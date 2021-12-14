import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from src.model.default_config import ModelConfig


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        d_model = config.d_model
        n_heads = config.fft_n_heads

        assert d_model % n_heads == 0
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.scale = torch.tensor(self.d_k ** (-0.5))

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        self.drop = nn.Dropout(config.p_dropout)

        self._init_weights()

    def forward(self, x, mask=None):
        """
        :param x: input sequence of shape [B, N, d_model] (we only need one sequence for self-attention)
        :param mask: boolean padding mask of shape [B, N] (True where to mask)
        :return: sequence of the same shape as x after self-attention
        """
        bsize = x.size(0)
        seq_len = x.size(1)
        assert x.size(2) == self.d_model

        q = self.Q(x).view(bsize, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.K(x).view(bsize, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.V(x).view(bsize, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # we transpose to get [B, n_heads, seq_len, d_k] so that BMM returns [B, n_heads, seq_len, seq_len]
        # we also use boolean mask here (-inf on masked positions)

        energy = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            energy.masked_fill_(mask[:, None, None, :], float('-inf'))

        weights = F.softmax(energy, dim=-1)
        attention = torch.matmul(self.drop(weights), v)\
                         .transpose(1, 2)\
                         .reshape(bsize, seq_len, self.d_model)

        return self.out(attention)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.Q.weight, gain=1.0)
        nn.init.xavier_uniform_(self.K.weight, gain=1.0)
        nn.init.xavier_uniform_(self.V.weight, gain=1.0)


class FFTBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        d_model = config.d_model
        p_dropout = config.p_dropout
        d_conv = config.fft_d_conv
        kernel = config.fft_kernel
        if type(kernel) is int:
            kernel = (kernel, kernel)
        pads = [k // 2 for k in kernel]

        self.mha = MultiHeadSelfAttention(config)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_conv, (kernel[0],), padding=pads[0]),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Conv1d(d_conv, d_model, (kernel[1],), padding=pads[1])
        )
        self.norm_first = config.fft_norm_first
        self.ln_mha = nn.LayerNorm(d_model)
        self.ln_conv = nn.LayerNorm(d_model)
        self.drop_mha = nn.Dropout(p_dropout)
        self.drop_conv = nn.Dropout(p_dropout)

    def forward(self, x, mask=None):
        """
        :param x: input sequence of shape [B, N, d_model]
        :param mask: boolean padding mask of shape [B, N] (True where to mask)
        :return: sequence of the same shape as x after FFTBlock
        """
        # norm_first is the only thing i checked in torch docs
        # (https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer)
        # This is done to make model more configurable
        if self.norm_first:
            x = x + self.drop_mha(self.mha(self.ln_mha(x), mask))
            x = x + self.drop_conv(self.conv(
                self.ln_conv(x).transpose(-1, -2)).transpose(-1, -2)
            )
        else:
            x = self.ln_mha(x + self.drop_mha(self.mha(x), mask))
            x = self.ln_conv(x + self.drop_conv(
                self.conv(x.transpose(-1, -2)).transpose(-1, -2)
            ))
        return x


class LengthRegulator(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        d_model = config.d_model
        d_conv = config.duration_d_conv
        p_dropout = config.p_dropout

        kernel = config.duration_kernel
        if type(kernel) is int:
            kernel = (kernel, kernel)
        pad = [k // 2 for k in kernel]

        self.conv1 = nn.Conv1d(d_model, d_conv, (kernel[0],), padding=pad[0])
        self.conv2 = nn.Conv1d(d_conv, d_conv, (kernel[1],), padding=pad[1])

        self.ln1 = nn.LayerNorm(d_conv)
        self.ln2 = nn.LayerNorm(d_conv)

        self.drop1 = nn.Dropout(p_dropout)
        self.drop2 = nn.Dropout(p_dropout)
        self.predictor = nn.Linear(d_conv, 1)
        self.alpha = config.duration_alpha

    def forward(self, x, true_durations=None):
        """
        :param x: input sequence of shape [B, N, d_model]
        :param true_durations: absolute floats predicted by aligner, will be used in train
        :return: output sequence of shape [B, N_new, d_model]
        :return: predicted log(lengths) of shape [B, N]
        """
        pred_lens = self.drop1(F.relu(self.ln1(
            self.conv1(x.transpose(-1, -2)).transpose(-1, -2)
        )))
        pred_lens = self.drop2(F.relu(self.ln2(
            self.conv2(pred_lens.transpose(-1, -2)).transpose(-1, -2)
        )))
        pred_lens = self.predictor(pred_lens).squeeze(-1)
        # pred_lens is [B, N]

        durations = true_durations if true_durations is not None else pred_lens.detach().exp()
        durations = torch.round(durations * self.alpha).to(torch.long)
        total_lens = durations.sum(-1)

        return self._create_extended_sequence(x, durations), pred_lens, total_lens

    @staticmethod
    def _create_extended_sequence(old_seq, durations):
        """
        :param old_seq: sequence to be extended, shape is [B, N, d_model]
        :param durations: durations of each element, shape is [B, N]
        :return: new_seq: extended sequence of shape [B, N_new, d_model]
        """
        new_seq = []
        for seq, dur in zip(old_seq, durations):
            new_seq.append(torch.repeat_interleave(seq, dur, dim=-2))
        return pad_sequence(new_seq, batch_first=True)

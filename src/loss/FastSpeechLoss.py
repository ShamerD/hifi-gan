import torch
from torch import nn

from src.utils.data import Batch


class FastSpeechLoss(nn.Module):
    def __init__(
            self,
            mel_loss=nn.MSELoss(),
            dur_loss=nn.MSELoss(),
    ):
        super().__init__()
        self.mel_loss = mel_loss
        self.dur_loss = dur_loss

    def forward(self, batch: Batch):
        assert batch.durations.size() == batch.durations_pred.size()
        batch_size = batch.durations.size(0)

        dur_loss = self.dur_loss(batch.durations, batch.durations_pred)

        mel_lens = torch.minimum(batch.mels_length, batch.mels_pred_length)

        mel_losses = torch.zeros(batch_size)
        for i in range(batch_size):
            mel_losses[i] = self.mel_loss(batch.mels[:, :, :mel_lens[i]], batch.mels_pred[:, :, :mel_lens[i]])
        mel_loss = mel_losses.mean()

        return mel_loss, dur_loss

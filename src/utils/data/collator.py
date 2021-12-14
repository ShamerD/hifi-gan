from dataclasses import dataclass, fields
from typing import Tuple, Optional, List

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class Batch:
    waveform: Optional[torch.Tensor]                    # [B, L]
    waveform_length: Optional[torch.Tensor]             # [B]

    transcript: List[str]
    tokens: torch.Tensor                                # [B, N]
    token_lengths: torch.Tensor                         # [B]

    durations: Optional[torch.Tensor] = None            # [B, N]
    durations_pred: Optional[torch.Tensor] = None       # [B, N]

    mels: Optional[torch.Tensor] = None                 # [B, n_mels, T]
    mels_pred: Optional[torch.Tensor] = None            # [B, n_mels, T']
    mels_length: Optional[torch.Tensor] = None          # [B]
    mels_pred_length: Optional[torch.Tensor] = None     # [B]

    mel_loss: Optional[torch.Tensor] = None
    dur_loss: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> 'Batch':
        for field in fields(self):
            if field.name in ["mel_loss", "dur_loss", "loss", "transcript"]:
                continue
            if field.name in ["tokens", "token_lengths"]:
                setattr(self, field.name, getattr(self, field.name).to(device))
                continue
            attr = getattr(self, field.name)
            if attr is not None:
                setattr(self, field.name, attr.to(device))

        return self


class LJSpeechCollator:

    def __call__(self, instances: List[Tuple]) -> Batch:
        waveform, waveform_length, transcript, tokens, token_lengths = list(
            zip(*instances)
        )

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)
        waveform_length = torch.cat(waveform_length)

        tokens = pad_sequence([
            tokens_[0] for tokens_ in tokens
        ]).transpose(0, 1)
        token_lengths = torch.cat(token_lengths)

        return Batch(waveform, waveform_length, transcript, tokens, token_lengths)


class TextCollator:
    def __call__(self, instances: List[Tuple]) -> Batch:
        transcript, tokens, token_lengths = list(
            zip(*instances)
        )

        tokens = pad_sequence([
            tokens_[0] for tokens_ in tokens
        ]).transpose(0, 1)
        token_lengths = torch.cat(token_lengths)

        return Batch(None, None, transcript, tokens, token_lengths)

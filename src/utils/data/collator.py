from dataclasses import dataclass, fields
from typing import Optional, List

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class Batch:
    wav: Optional[torch.Tensor] = None          # [B, L]
    wav_gen: Optional[torch.Tensor] = None      # [B, L']

    spec: Optional[torch.Tensor] = None         # [B, n_mels, T]
    spec_gen: Optional[torch.Tensor] = None     # [B, n_mels, T]

    def to(self, device: torch.device) -> 'Batch':
        for field in fields(self):
            attr = getattr(self, field.name)
            if attr is not None:
                setattr(self, field.name, attr.to(device))

        return self


class LJSpeechCollator:
    def __call__(self, instances: List) -> Batch:
        waveform = instances

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)

        return Batch(waveform)

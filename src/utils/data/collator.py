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

    # these contain outputs of every discriminator sub-block
    disc_pred: Optional[List[torch.Tensor]] = None
    disc_pred_gen: Optional[List[torch.Tensor]] = None

    # these contain outputs of every discriminator layer
    disc_features: Optional[List[torch.Tensor]] = None
    disc_features_gen: Optional[List[torch.Tensor]] = None

    # we put losses here to simplify its calculation and logging
    # generator losses
    generator_loss: Optional[torch.Tensor] = None
    spec_loss: Optional[torch.Tensor] = None
    adv_gen_loss: Optional[torch.Tensor] = None
    feature_loss: Optional[torch.Tensor] = None

    # discriminator losses
    discriminator_loss: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> 'Batch':
        for field in fields(self):
            if field.name.endswith('loss'):
                continue
            attr = getattr(self, field.name)
            if attr is None:
                continue
            if field.name.startswith('disc'):
                attr = [t.to(device) for t in attr]
            else:
                attr = attr.to(device)
            setattr(self, field.name, attr)

        return self


class LJSpeechCollator:
    def __call__(self, instances: List) -> Batch:
        waveform = instances

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)

        return Batch(waveform)

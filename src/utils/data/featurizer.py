from dataclasses import dataclass

import librosa
import torch
import torchaudio
from torch import nn


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0

    # if no padding is applied mel length should be (N - window + hop) // hop, if N = wav_length
    # but center=True (default) pads left and right by (window // 2), so mel_len = (N + hop) // hop
    # we want it to be exactly N / hop (as model upsamples it x256 (hop))
    # we can choose N to be divisible by hop, but we carefully choose padding to make mel_length = (N / hop)
    # that being said padding = (window - hop) / 2

    center: bool = False
    pad: int = (1024 - 256) // 2

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251


class MelSpectrogram(nn.Module):
    def __init__(self, config: MelSpectrogramConfig):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            power=config.power,
            center=config.center,
            pad=config.pad
        )

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: log-mel-spectrogram of shape [B, n_mels, T']
        """

        # by default audio time is divisible by hop_length
        # because of that mel_time will be T / hop_length + 1
        # model's output will be hop_length * mel_time = T + hop_length
        # that is if we calculate spec again we will be 1 off real spec in mel_time

        mel = self.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

        return mel

    def transform_wav_lengths(self, wav_lengths: torch.Tensor):
        # if no padding this should be (N - window + hop) // hop
        # but we chose padding to make it N // hop
        return torch.div(
            wav_lengths,
            self.config.hop_length,
            rounding_mode='trunc'
        )

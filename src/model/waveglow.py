import warnings

import torch
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch import nn

from src.utils import CHECKPOINT_DIR


class Waveglow(nn.Module):
    GDRIVE_FILE_ID = '1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF'

    def __init__(self, checkpoint='waveglow_256channels_universal_v5.pt'):
        super().__init__()
        self.sr = 22050

        model_path = CHECKPOINT_DIR / checkpoint
        if not model_path.exists():
            gdd.download_file_from_google_drive(
                file_id=self.GDRIVE_FILE_ID,
                dest_path=str(model_path)
            )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = torch.load(model_path, map_location='cpu')['model']
            self.net = model.remove_weightnorm(model)

    @torch.no_grad()
    def inference(self, spec: torch.Tensor):
        spec = self.net.upsample(spec)

        # trim the conv artifacts
        time_cutoff = self.net.upsample.kernel_size[0] - \
            self.net.upsample.stride[0]
        spec = spec[:, :, :-time_cutoff]

        spec = spec.unfold(2, self.net.n_group, self.net.n_group) \
            .permute(0, 2, 1, 3) \
            .contiguous() \
            .flatten(start_dim=2) \
            .transpose(-1, -2)

        # generate prior
        audio = torch.randn(spec.size(0), self.net.n_remaining_channels, spec.size(-1)) \
            .to(spec.device)

        for k in reversed(range(self.net.n_flows)):
            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.net.WN[k]((audio_0, spec))

            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b) / torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)

            audio = self.net.convinv[k](audio, reverse=True)

            if k % self.net.n_early_every == 0 and k > 0:
                z = torch.randn(
                    spec.size(0), self.net.n_early_size, spec.size(2),
                    device=spec.device
                )
                audio = torch.cat((z, audio), 1)

        audio = audio.permute(0, 2, 1) \
            .contiguous() \
            .view(audio.size(0), -1)

        return audio

import torch
from torch import nn
import torch.nn.functional as F

from src.model.default_config import ModelConfig

"""
All discriminator layers return intermediate feature maps
"""


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, config: ModelConfig):
        super().__init__()

        self.period = period
        self.layers = nn.ModuleList()

        channels = [1] + [config.mpd_sublayer_base_channels * (2 ** i) for i in range(config.mpd_num_sublayers + 1)]
        kernel_size = config.mpd_sublayer_kernel_size

        # add repeated convs and the one with stride=1
        for i in range(len(channels) - 1):
            is_last = (i + 2 == len(channels))
            self.layers.append(
                nn.Conv2d(channels[i], channels[i+1],
                          (kernel_size, 1),
                          (config.mpd_sublayer_stride if not is_last else 1, 1),
                          padding=((kernel_size - 1) // 2, 0))
            )
            self.layers.append(nn.LeakyReLU(config.relu_slope))

        # add last layer making channels=1, note no relu
        last_ks = config.mpd_sublayer_last_kernel_size
        self.layers.append(
            nn.Conv2d(channels[-1], 1,
                      kernel_size=(last_ks, 1),
                      padding=((last_ks - 1) // 2, 0))
        )

        self.flat_fn = nn.Flatten()

    def forward(self, x: torch.Tensor):
        intermediate = []

        # pad and reshape
        batch_size, time = x.size()
        if time % self.period != 0:
            need_to_add = self.period - (time % self.period)
            x = F.pad(x, (0, need_to_add))
            time += need_to_add

        x = x.reshape(batch_size, 1, time // self.period, self.period)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            # we save every time after relu and after the last conv layer
            if i % 2 == 1 or i + 1 == len(self.layers):
                intermediate.append(x.clone())

        x = self.flat_fn(x)

        return x, intermediate


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p, config) for p in config.mpd_periods
        ])

    def forward(self, x):
        predictions = []
        intermediate = []

        for layer in self.discriminators:
            pred, features = layer(x)
            predictions.append(pred)
            intermediate.extend(features)

        return predictions, intermediate


class ScaleDiscriminator(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.n_convs = len(config.msd_sublayer_channels)
        self.layers = nn.ModuleList()

        last_n_channels = 1
        for idx, (n_channels, kernel_size, stride, groups) in enumerate(zip(config.msd_sublayer_channels,
                                                                            config.msd_sublayer_kernels,
                                                                            config.msd_sublayer_strides,
                                                                            config.msd_sublayer_groups)):
            self.layers.append(nn.Conv1d(last_n_channels, n_channels,
                                         kernel_size, stride,
                                         padding=((kernel_size - 1) // 2),
                                         groups=groups))

            if idx + 1 < self.n_convs:
                self.layers.append(nn.LeakyReLU(config.relu_slope))
            last_n_channels = n_channels

        self.flat_fn = nn.Flatten()

    def forward(self, x: torch.Tensor):
        intermediate = []

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i % 2 == 1 or i + 1 == len(self.layers):
                intermediate.append(x.clone())

        x = self.flat_fn(x)
        return x, intermediate


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.n_layers = config.msd_num_layers
        self.discriminators = nn.ModuleList(ScaleDiscriminator(config) for _ in range(self.n_layers))
        self.pool = nn.AvgPool1d(4, 2, 2)

    def forward(self, x: torch.Tensor):
        predictions = []
        intermediate = []
        x = x.unsqueeze(1)

        for i in range(self.n_layers):
            pred, features = self.discriminators[i](x)
            if i + 1 < self.n_layers:
                x = self.pool(x)

            predictions.append(pred)
            intermediate.extend(features)

        return predictions, intermediate


class HiFiDiscriminator(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(config)
        self.msd = MultiScaleDiscriminator(config)

    def forward(self, x: torch.Tensor):
        """
        :param x: wavs of shape [B, L]
        :return: (list of all discriminator predictions, list of all discriminators' intermediate features)
        """
        mpd_pred, mpd_feats = self.mpd(x)
        msd_pred, msd_feats = self.msd(x)

        return mpd_pred + msd_pred, mpd_feats + msd_feats

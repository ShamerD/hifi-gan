import torch
from torch import nn

from src.model.default_config import ModelConfig


class ResBlock(nn.Module):
    def __init__(self, block_id: int, n_channels: int, config: ModelConfig):
        super().__init__()

        kernel_size = config.kernels_resblocks[block_id]
        dilations = config.dilations_resblocks[block_id]

        self.num_outer = len(dilations)
        self.num_inner = len(dilations[0])
        self.kernel_size = kernel_size

        layers = []

        for d_outer in dilations:
            inner_layers = []
            for d_inner in d_outer:
                inner_layers.extend([
                    nn.LeakyReLU(config.relu_slope),
                    nn.Conv1d(n_channels, n_channels, kernel_size, stride=1, padding='same', dilation=d_inner)
                ])
            layers.append(nn.Sequential(*inner_layers))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)

        return x


class MRFBlock(nn.Module):
    def __init__(self, n_channels: int, config: ModelConfig):
        super().__init__()

        self.layers = nn.ModuleList()
        self.n_blocks = len(config.kernels_resblocks)

        for i in range(self.n_blocks):
            self.layers.append(ResBlock(i, n_channels, config))

    def forward(self, x):
        out = 0
        for layer in self.layers:
            out += layer(x)
        return out


class GeneratorBlock(nn.Module):
    def __init__(self, block_id: int, in_channels: int, out_channels: int, config: ModelConfig):
        super().__init__()
        kernel_size = config.kernels_upsample[block_id]

        self.net = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                               stride=(kernel_size // 2), padding=(kernel_size // 4)),
            MRFBlock(out_channels, config),
            nn.LeakyReLU(config.relu_slope)
        )

    def forward(self, x):
        return self.net(x)


class HiFiGenerator(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_blocks = len(config.kernels_upsample)
        self.n_resblocks = len(config.kernels_resblocks)

        self.pre = nn.Sequential(
            nn.Conv1d(config.n_mels, config.d_hidden, config.pre_kernel_size, padding='same'),
            nn.LeakyReLU(config.relu_slope)
        )

        # every block upsamples time dimension and reduces channel dimension (x2)
        layers = []
        for i in range(self.n_blocks):
            d_block = config.d_hidden // (2 ** i)
            layers.append(GeneratorBlock(i, d_block, d_block // 2, config))
        self.net = nn.Sequential(*layers)

        self.post = nn.Sequential(
            nn.Conv1d(config.d_hidden // (2 ** self.n_blocks), 1, config.post_kernel_size, padding='same'),
            nn.Flatten(),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: melspecs of shape [B, n_mels, T]
        :return: wavs of shape [B, L]
        """
        return self.post(self.net(self.pre(x)))

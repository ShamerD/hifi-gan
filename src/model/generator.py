import torch
from torch import nn
from torch.nn.utils import weight_norm

from src.model.default_config import ModelConfig
from src.utils import init_normal


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
                inner_layers.append(nn.LeakyReLU(config.relu_slope))

                conv = nn.Conv1d(n_channels, n_channels, kernel_size, stride=1, padding='same', dilation=d_inner)
                init_normal(conv)
                conv = weight_norm(conv)

                inner_layers.append(conv)
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

        upsample_conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                           stride=(kernel_size // 2), padding=(kernel_size // 4))
        init_normal(upsample_conv)
        upsample_conv = weight_norm(upsample_conv)

        self.net = nn.Sequential(
            upsample_conv,
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

        pre_conv = nn.Conv1d(config.n_mels, config.d_hidden, config.pre_kernel_size, padding='same')
        init_normal(pre_conv)
        pre_conv = weight_norm(pre_conv)

        self.pre = nn.Sequential(
            pre_conv,
            nn.LeakyReLU(config.relu_slope)
        )

        # every block upsamples time dimension and reduces channel dimension (x2)
        layers = []
        for i in range(self.n_blocks):
            d_block = config.d_hidden // (2 ** i)
            layers.append(GeneratorBlock(i, d_block, d_block // 2, config))
        self.net = nn.Sequential(*layers)

        post_conv = nn.Conv1d(config.d_hidden // (2 ** self.n_blocks), 1, config.post_kernel_size, padding='same')
        init_normal(post_conv)
        post_conv = weight_norm(post_conv)

        self.post = nn.Sequential(
            post_conv,
            nn.Flatten(),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: melspecs of shape [B, n_mels, T]
        :return: wavs of shape [B, L]
        """
        return self.post(self.net(self.pre(x)))

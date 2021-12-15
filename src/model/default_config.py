from dataclasses import dataclass, field
from typing import Union, Tuple, List


@dataclass
class ModelConfig:
    # These are default parameters that can be changed via json config
    d_hidden: int = 512
    n_mels: int = 80

    relu_slope: float = 0.1

    # Generator
    pre_kernel_size: int = 7
    post_kernel_size: int = 7
    kernels_upsample: Tuple[int] = (16, 16, 4, 4)
    kernels_resblocks: Tuple[int] = (3, 7, 11)
    dilations_resblocks: Tuple[Tuple[Tuple[int]]] = (((1, 1), (3, 1), (5, 1)),) * 3

    # Discriminator
    mpd_periods: Tuple[int] = (2, 3, 5, 7, 11)
    mpd_num_sublayers: int = 4
    mpd_sublayer_base_channels: int = 64
    mpd_sublayer_kernel_size: int = 5
    mpd_sublayer_last_kernel_size: int = 3
    mpd_sublayer_stride: int = 3

    msd_num_layers: int = 3
    msd_sublayer_channels: Tuple[int] = (16, 64, 256, 1024, 1024, 1024, 1)
    msd_sublayer_kernels: Tuple[int] = (15, 41, 41, 41, 41, 5, 3)
    msd_sublayer_strides: Tuple[int] = (1, 4, 4, 4, 4, 1, 1)
    msd_sublayer_groups: Tuple[int] = (1, 4, 16, 64, 256, 1, 1)

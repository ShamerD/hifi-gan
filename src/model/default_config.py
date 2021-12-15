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
    kernels_upsample: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    kernels_resblocks: List[int] = field(default_factory=lambda: [3, 7, 11])
    dilations_resblocks: List[List[List[int]]] = field(default_factory=lambda: [[[1, 1], [3, 1], [5, 1]]] * 3)

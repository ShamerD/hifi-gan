from dataclasses import dataclass
from typing import Union, Tuple


@dataclass
class ModelConfig:
    # These are default parameters that can be changed via json config
    d_model: int = 384
    n_mels: int = 80

    n_encoder_layers: int = 6
    n_decoder_layers: int = 6

    p_dropout: float = 0.1

    # Duration predictor
    duration_kernel: Union[int, Tuple[int, int]] = 3
    duration_d_conv: int = 256
    duration_alpha: float = 1.0

    # FFTBlock
    fft_norm_first: bool = True
    fft_d_conv: int = 1536
    fft_kernel: Union[int, Tuple[int, int]] = 3
    fft_n_heads: int = 2

    # Embeddings
    enc_vocab_size: int = 38
    enc_pos_size: int = 1024  # TODO: check on that + maybe nontrainable
    dec_pos_size: int = 2048

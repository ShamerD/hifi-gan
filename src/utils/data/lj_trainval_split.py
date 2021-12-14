from typing import Union

import numpy as np

from src.utils import DATA_DIR, fix_seed

LJ_SIZE = 13100
LJ_DATA_DIR = DATA_DIR / "LJSpeech-1.1"


def generate_trainval_split(val_size: Union[float, int]):
    if type(val_size) == float:
        val_size = int(LJ_SIZE * val_size)

    indices = np.arange(LJ_SIZE)

    fix_seed()
    np.random.shuffle(indices)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    np.save(LJ_DATA_DIR / "train", train_indices)
    np.save(LJ_DATA_DIR / "val", val_indices)


if __name__ == "__main__":
    generate_trainval_split(600)

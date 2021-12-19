import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio

from src.utils import DATA_DIR


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self, max_wav_length: Optional[int] = None, deterministic: bool = False):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        super().__init__(root=DATA_DIR, download=True)
        self.max_wav_length = max_wav_length
        self.deterministic = deterministic

    def __getitem__(self, index: int):
        waveform, _, _, _ = super().__getitem__(index)

        # 0dim is for channels
        if self.max_wav_length is not None and waveform.size(1) > self.max_wav_length:
            max_start = waveform.size(1) - self.max_wav_length
            if self.deterministic:
                start = min(max_start, waveform.size(1) // 2)
            else:
                start = random.randint(0, max_start)
            waveform = waveform[:, start:(start + self.max_wav_length)]

        return waveform


class InferenceWAVDataset(Dataset):
    def __init__(self, data_dir=(DATA_DIR / "default_example")):
        """
        Dataset where each entry is wav (is needed to be converted to mel)
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.filenames = os.listdir(data_dir)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.data_dir / self.filenames[idx])
        return waveform

    def __len__(self):
        return len(self.filenames)


class InferenceMelDataset(Dataset):
    def __init__(self, data_dir):
        """
        Dataset where each entry is already melspec (in .npy format)
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.filenames = os.listdir(data_dir)

    def __getitem__(self, idx):
        waveform = np.load(str(self.data_dir / self.filenames[idx]))
        return torch.from_numpy(waveform)

    def __len__(self):
        return len(self.filenames)

    def get_item_path(self, idx):
        return str(self.data_dir / self.filenames[idx])


if __name__ == "__main__":
    # download check
    lj_speech = LJSpeechDataset()
    print(lj_speech[0])

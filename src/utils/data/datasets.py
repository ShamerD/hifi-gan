import random
from typing import Optional

import torchaudio

from src.utils import DATA_DIR


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self, max_wav_length: Optional[int] = None):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        super().__init__(root=DATA_DIR, download=True)
        self.max_wav_length = max_wav_length

    def __getitem__(self, index: int):
        waveform, _, _, _ = super().__getitem__(index)

        # 0dim is for channels
        if self.max_wav_length is not None and waveform.size(1) > self.max_wav_length:
            max_start = waveform.size(1) - self.max_wav_length
            start = random.randint(0, max_start)
            waveform = waveform[:, start:(start + self.max_wav_length)]

        return waveform


if __name__ == "__main__":
    # download check
    lj_speech = LJSpeechDataset()
    print(lj_speech[0])

import torchaudio

from src.utils import DATA_DIR


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        super().__init__(root=DATA_DIR, download=True)

    def __getitem__(self, index: int):
        waveform, _, _, _ = super().__getitem__(index)

        return waveform


if __name__ == "__main__":
    # download check
    lj_speech = LJSpeechDataset()
    print(lj_speech[0])

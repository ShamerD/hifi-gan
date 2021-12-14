import unicodedata
from typing import List, Optional

import torch
import torchaudio
from torch.utils.data import Dataset

from src.utils import DATA_DIR


def remove_accents(text: str):
    """
    :param text: single text possibly containing non-ascii symbols (eg. Müller)
    :return: normalized text with accents removed

    LJSpeech dataset contains 19 entries with accented text.
    They can cause trouble, because 2 tokenizers (dataset's and aligner's) handle accented texts differently:
    dataset's tokenizer simply ignores them, aligner's treat them as <unk>.
    This leads to wrong alignment if not runtime error (if entry is batch's longest text)
    """
    return "".join([c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)])


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        super().__init__(root=DATA_DIR, download=True)
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        waveform_length = torch.tensor([waveform.shape[-1]]).int()

        transcript = remove_accents(transcript)
        tokens, token_lengths = self._tokenizer(transcript)
        transcript = self.decode(tokens, token_lengths)[0]

        return waveform, waveform_length, transcript, tokens, token_lengths

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result


TEXT_EXAMPLES = [
    "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
    "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
    "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space"
]


class TextDataset(Dataset):
    def __init__(self, texts: Optional[List[str]] = None):
        super().__init__()
        self.texts = texts if texts is not None else TEXT_EXAMPLES
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

    def __getitem__(self, index):
        transcript = self.texts[index]
        transcript = remove_accents(transcript)
        tokens, token_lengths = self._tokenizer(transcript)
        transcript = self.decode(tokens, token_lengths)[0]

        return transcript, tokens, token_lengths

    def __len__(self):
        return len(self.texts)

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result


if __name__ == "__main__":
    # download check
    lj_speech = LJSpeechDataset()
    print(lj_speech[0])

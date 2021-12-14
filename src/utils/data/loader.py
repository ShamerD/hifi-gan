from typing import Dict

import numpy as np
from numpy.random import shuffle
from torch.utils.data import DataLoader, Subset

from src.utils.data import LJSpeechDataset, LJSpeechCollator, TextDataset, TextCollator


def get_dataloaders(data_config: Dict):
    dataloaders = {'train': None, 'val': None, 'inference': None}

    batch_size = data_config.get('batch_size', 1)
    num_workers = data_config.get('num_workers', 1)

    dataset = LJSpeechDataset()
    val_dataset = None

    if 'val_path' in data_config and 'train_path' in data_config:
        val_idx = np.load(data_config['val_path'])
        train_idx = np.load(data_config['train_path'])

        val_dataset = Subset(dataset, val_idx)
        train_dataset = Subset(dataset, train_idx)
    else:
        train_dataset = dataset

    if 'limit' in data_config:
        idx = list(range(len(train_dataset)))
        shuffle(idx)
        idx = idx[:data_config['limit']]

        train_dataset = Subset(dataset, idx)

    dataloaders['train'] = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      collate_fn=LJSpeechCollator(),
                                      shuffle=True,
                                      num_workers=num_workers)

    if val_dataset is not None:
        dataloaders['val'] = DataLoader(val_dataset,
                                        batch_size=batch_size,
                                        collate_fn=LJSpeechCollator(),
                                        shuffle=False,
                                        num_workers=num_workers)

    inference_dataset = TextDataset()
    dataloaders['inference'] = DataLoader(inference_dataset,
                                          batch_size=1,
                                          collate_fn=TextCollator(),
                                          shuffle=False)

    return dataloaders

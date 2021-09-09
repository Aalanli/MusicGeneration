# %%
import math
import os
import random
import _pickle as pickle

import torch
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, datadir, seq_len) -> None:
        self.data_files = [f'{datadir}/{file}' for file in os.listdir(datadir)]
        random.shuffle(self.data_files)
        self.end = len(self.data_files)

        self.seq_len = seq_len
    
    def __len__(self):
        return self.end
    
    def __getitem__(self, index):
        file = self.data_files[index]
        with open(file, 'rb') as f:
            # a 1D list containing integers
            sequence = pickle.load(f)
        # the first 0 class for padding, shift every encoding by 1
        sequence = torch.tensor(sequence) + 1
        sequence = sequence.flatten(0)
        if sequence.shape[0] < self.seq_len:
            return torch.nn.functional.pad(sequence, (0, self.seq_len - sequence.shape[0]))
        
        window = sequence.shape[0] - self.seq_len
        start = random.randint(0, window)
        return sequence[start:start + self.seq_len]

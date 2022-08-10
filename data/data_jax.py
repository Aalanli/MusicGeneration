# %%
import math
import os
import random
import _pickle as pickle # cpickle, faster maybe

from collections import deque

import jax
from jax import numpy as jnp
import numpy as np
import ray


@ray.remote
def process(file, target_len):
    with open(file, 'rb') as f:
        # a 1D list containing integers
        sequence = pickle.load(f)
    # the first 0 class for [start] token, 1 class for [end] token
    # shift every encoding by 2
    len_seq = len(sequence)
    if len_seq + 1 < target_len:
        sequence = np.asarray(sequence, np.int32) + 2
        arr = np.ones([target_len], np.int32)
        arr[0] = 0
        arr[1:len_seq+1] = sequence
        sequence = arr
    else:
        start = random.randint(0, len_seq - target_len - 2)
        arr = np.ones([target_len], np.int32)
        arr[1:] = np.asarray(sequence[start:start+target_len-1], np.int32) + 2
        arr[0] = 0
        sequence = arr
    return sequence
    
class Data:
    def __init__(self, datadir, seq_len, batch_size, queue_size=4) -> None:
        self.data_files = [f'{datadir}/{file}' for file in os.listdir(datadir)]
        random.shuffle(self.data_files)
        self.end = len(self.data_files)
        self.batch_size = batch_size

        self.seq_len = seq_len
        self.idx = queue_size
        self.queue = deque([process.remote(self.data_files[i], seq_len) for i in range(0, queue_size * batch_size)])
    
    def __len__(self):
        return self.end
    
    def get(self):
        values = [ray.get(self.queue.pop()) for i in range(self.batch_size)]
        for i in range(self.batch_size): 
            self.queue.append(process.remote(self.data_files[(i + self.idx) % self.end], self.seq_len))
        self.idx += self.batch_size
        return jnp.array(values)


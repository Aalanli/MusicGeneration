# %%
import random
from typing import Iterator
import ray
import os
import pickle
import numpy as np
from pathlib import Path
from itertools import cycle
from collections import deque

import data.parse_midi_numba as pm


@ray.remote
def file_to_midi_norm_stream(file: str):
    raw_stream = pm.parse_raw_midi_stream(file)
    stream = pm.raw_stream_to_stream(raw_stream)
    return stream


def get_midi_files(data_dir):
    data_dir = Path(data_dir)
    file_paths = sorted(data_dir.glob('**/*.midi'))
    file_paths.extend(sorted(data_dir.glob('**/*.mid')))
    return [str(i) for i in file_paths]


class Dataset:
    """
    Each actor takes a list of data, transforms and batches it into the final form of what the model is expecting.
    """
    def __init__(self,
        data_dir,
        batch_size,
        transform_actors,
        files=None,
        cache_file=None) -> None:
        
        self.batch_size = batch_size
        self.actors = transform_actors
        self.cache_file = cache_file

        if self.cache_file is not None and os.path.exists(cache_file):
            with open(self.cache_file, 'rb') as f:
                self.data = pickle.load(f)
                self.obj_refs = []
        else:
            if files is not None:
                self.files = files
            else:
                self.files = get_midi_files(data_dir)
            random.shuffle(self.files)
            self.obj_refs = [file_to_midi_norm_stream.remote(f) for f in self.files]
            self.data = []

        self.obj_queue = deque()
    
    def __len__(self):
        return len(self.files)
    
    def flush_cache(self):
        if isinstance(self.data, Iterator):
            self.obj_queue.clear()
            for i in range(len(self.actors)):
                # save the worker id for each computation
                self.obj_queue.append((self.actors[i].call.remote([next(self.data) for _ in range(self.batch_size)]), i))
    
    def get(self):
        if len(self.obj_refs) > self.batch_size:
            data, self.obj_refs = ray.wait(self.obj_refs, num_returns=self.batch_size)
            data = ray.get(data)
            self.data.extend(data)
            x, y = ray.get(self.actors[0].call.remote(data))
            return x, y
        elif isinstance(self.data, list):
            # fixes bug if number of files is not a multiple of batch_size
            if 0 < len(self.obj_refs) and len(self.obj_refs) < self.batch_size:
                data = ray.get(self.obj_refs)
                self.obj_refs = []
                self.data.extend(data)
            # now that everything is computed, cache data if option is set
            if self.cache_file is not None and not os.path.exists(self.cache_file):
                if not os.path.exists(os.path.dirname(self.cache_file)):
                    os.makedirs(os.path.dirname(self.cache_file))
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.data, f)
            
            self.data = cycle(self.data)
            # prepare a queue of waiting computation, such that len(queue) = self.workers
            self.flush_cache()
        
        w_ref, w_id = self.obj_queue.popleft()
        x, y = ray.get(w_ref)
        # make the most recent finished worker to more work
        self.obj_queue.append((self.actors[w_id].call.remote([next(self.data) for _ in range(self.batch_size)]) , w_id))
        return x, y


def build(args):
    return Dataset(**args, workers=2)


if __name__ == '__main__':
    from misc import EasyDict
    from data.datasetv2 import Dataset
    from data.transforms import SeparatedEncoding, UnifiedEncoding, UnifiedReconstruct

    d_args = EasyDict()

    data_dir = r'datasets/maestro/maestro-v3.0.0'
    d_args.batch_size            = 4
    d_args.workers               = 2
    d_args.seq_len               = 1024
    d_args.duration_lin_bins     = 200
    d_args.note_shifts           = (-5, 5)
    d_args.velocity_shifts       = (-7, 7)
    d_args.duration_muls         = (0.8, 1.2)

    actors = [UnifiedEncoding.remote(
        d_args.seq_len,
        d_args.note_shifts,
        d_args.velocity_shifts,
        d_args.duration_muls,
        d_args.duration_lin_bins
    ) for _ in range(d_args.workers)]
    dataset = Dataset(data_dir, d_args.batch_size, actors)
    # %%
    x, y = dataset.get()
    rc = UnifiedReconstruct()
    rc.binned_encoding_to_file('reconstructed.midi', x[0])


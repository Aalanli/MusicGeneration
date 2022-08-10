# %%
from dataclasses import dataclass
import random
from typing import Tuple
import ray
import numpy as np
from pathlib import Path
from itertools import cycle
import copy

import data.parse_midi_numba as pm


# %%
@ray.remote
def file_to_midi_norm_stream(file: str):
    raw_stream = pm.parse_raw_midi_stream(file)
    stream = pm.raw_stream_to_stream(raw_stream)
    durations = stream.events[:, 2] / stream.ticks
    notes = stream.events[:, 0] + 1
    return notes, stream.events[:, 1], durations


def get_midi_files(data_dir):
    data_dir = Path(data_dir)
    file_paths = sorted(data_dir.glob('**/*.midi'))
    file_paths.extend(sorted(data_dir.glob('**/*.mid')))
    return [str(i) for i in file_paths]

# %%
@ray.remote
@dataclass
class Transforms:
    seq_len: int
    notes: Tuple[int, int]
    velocities: Tuple[int, int]
    durations: Tuple[float, float]
    duration_lin_bins: int = 100
    note_bins: int = 89
    velocity_bins: int = 127

    def __post_init__(self):
        self.bin_arr = np.cumsum(pm.compute_lin_bin(0, 5, self.duration_lin_bins))

        self.duration_bins = self.duration_lin_bins
        # reserve 2 extra bins for beginning of sequence and end of sequence
        self.final_note_bins = self.note_bins + 2
        self.final_velocity_bins = self.velocity_bins + 2
        self.final_duration_bins = self.duration_bins + 2
    
    def pad(self, arr: np.ndarray):
        arr += 2
        pad = self.seq_len - arr.shape[0] - 2
        return np.concatenate([[0], arr, np.ones(1 + pad, dtype=arr.dtype)])
      
    def transform(self, notes: np.ndarray, velocities: np.ndarray, durations: np.ndarray) -> np.ndarray:
        assert notes.ndim == 1
        notes = copy.deepcopy(notes)
        velocities = copy.deepcopy(velocities)
        durations = copy.deepcopy(durations)

        # truncate to required size
        if notes.shape[0] > self.seq_len - 2:
            idx = np.random.randint(0, notes.shape[0] - self.seq_len - 2)
            end = idx + self.seq_len - 2
            notes = notes[idx:end]
            velocities = velocities[idx:end]
            durations = durations[idx:end]
        
        notes[np.where(notes != 0)] += np.random.randint(self.notes[0], self.notes[1])
        notes = np.clip(notes, 0, self.note_bins)
        notes = self.pad(notes)

        velocities = np.random.randint(self.velocities[0], self.velocities[1]) + velocities
        velocities = np.clip(velocities, 0, self.velocity_bins)
        velocities = self.pad(velocities)

        durations = durations * np.random.uniform(self.durations[0], self.durations[1])
        durations = np.digitize(durations, self.bin_arr)
        durations = np.clip(durations, 0, self.duration_bins)
        durations = self.pad(durations)

        return np.stack([notes, velocities, durations], axis=0)


@dataclass
class Reconstruct:
    ticks: int = 480
    duration_lin_bins: int = 100
    note_bins: int = 89
    velocity_bins: int = 127

    def __post_init__(self):
        self.bin_arr = np.cumsum(pm.compute_lin_bin(0, 5, self.duration_lin_bins))
    
    def reconstruct_stream(self, arr: np.ndarray) -> pm.MidiStream:
        """
        arr: the output format of Transforms.transform with shape (3, N)
        """
        # remove paddings, which is class 1 and 0
        arr = copy.deepcopy(arr)
        # remove padding
        if (arr == 0).any():
            arr = arr[:, 1:]
        if (arr == 1).any():
            ind = np.argmax(arr[0] == 1)
            arr = arr[:, :ind]

        arr[0] = np.clip(arr[0], 0, self.note_bins)
        arr[1] = np.clip(arr[1], 0, self.velocity_bins)
        arr[2] = np.clip(arr[2], 0, self.duration_lin_bins)

        arr -= 2 # unshift padding from transfroms 
        arr[0] -= 1 # unshift note from file_to_midi_norm_stream

        # remove notes that have a velocity of 0
        ind ,= np.where((arr[1] != 0) | (arr[0] == -1))
        arr = arr[:, ind]

        # unbin and scale durations
        arr[2] = (self.bin_arr[arr[2]] * self.ticks).astype(arr.dtype)
        arr = arr.transpose((1, 0))
        #arr = np.flip(arr, 0)
        return pm.MidiStream(self.ticks, arr)
    
    def binned_encoding_to_file(self, file_name: str, arr: np.ndarray) -> None:
        stream = self.reconstruct_stream(arr)
        raw_stream = pm.stream_to_raw_stream(stream)
        pm.raw_stream_to_midi_file(file_name, raw_stream)


class Dataset:
    def __init__(self,
        data_dir,
        batch_size,
        workers,
        seq_len,
        duration_lin_bins = 100,
        note_shifts = (-5, 5),
        velocity_shifts = (-7, 7),
        duration_muls = (0.8, 1.2)) -> None:
        
        self.batch_size = batch_size
        self.workers = workers
        self.actors = [Transforms.remote(seq_len, note_shifts, velocity_shifts, duration_muls, 
            duration_lin_bins) for _ in range(workers)]
        
        self.files = get_midi_files(data_dir)
        random.shuffle(self.files)
        self.obj_refs = [file_to_midi_norm_stream.remote(f) for f in self.files]
        self.data = []

        self.obj_store = []
    
    def switch_actors(self, actor, seq_len, note_shifts, velocity_shifts, duration_muls, dur_lin_bins):
        self.actors = [actor.remote(seq_len, note_shifts, velocity_shifts, duration_muls, 
            dur_lin_bins) for _ in range(self.workers)]
    
    def __len__(self):
        return len(self.files)
    
    def get(self):
        if len(self.obj_refs) > 0:
            data, self.obj_refs = ray.wait(self.obj_refs, num_returns=self.batch_size)
            data = ray.get(data)
            self.data.extend(data)
            arr = ray.get([self.actors[i % self.workers].transform.remote(*data[i]) for i in range(self.batch_size)])
            arr = np.stack(arr, axis=0)
            return [arr[:, :, :-1]], arr[:, :, 1:]
        elif isinstance(self.data, list):
            self.data = cycle(self.data)
            self.obj_store = [self.actors[i % self.workers].transform.remote(*next(self.data)) for i in range(self.batch_size * 3)]
        
        data, self.obj_store = ray.wait(self.obj_store, num_returns=self.batch_size)
        self.obj_store.extend([self.actors[i % self.workers].transform.remote(*next(self.data)) for i in range(self.batch_size)])
        data = np.stack(ray.get(data), axis=0)
        return [data[:, :, :-1]], data[:, :, 1:]

def build(args):
    return Dataset(**args, workers=2)

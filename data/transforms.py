# %%
from dataclasses import dataclass
from typing import List, Tuple
import copy

import ray
import numpy as np
import data.parse_midi_numba as pm


@ray.remote
@dataclass
class SeparatedEncoding:
    """
    Separate the encodings/embeddings for notes, velocities and durations,
    such that a single embedding vector is constructed from the concatenation
    of the embeddings from the three, resulting in more permutations from fewer
    embeddings. This may be desirable for tasks where qualities of embeddings
    could be clearly separated.
    """
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
      
    def transform_single_batch(self, notes: np.ndarray, velocities: np.ndarray, durations: np.ndarray) -> np.ndarray:
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
    
    def call(self, data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        data = np.stack([self.transform_single_batch(*d) for d in data], axis=0)
        return data[:, :, :-1], data[:, :, 1:]


@dataclass
class SeparatedReconstruct:
    """
    Reconstruct the midi stream from the outputs of SeparatedEncoding
    """
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
# %%
from dataclasses import dataclass
from typing import List, Tuple
import copy

import ray
import numpy as np
import data.parse_midi_numba as pm
import numba as nb

@nb.njit
def flatten_notes(notes, velocities, durations, note_bins=89, velocity_bins=127, duration_bins=200):
    """
    Flattens bins such that:
    Following every note: (note: 0 -> note_bins-1), (velocity: prev -> prev + note_bins+velocity_bins-1), (duration_n_bin: prev -> prev + note_bins+velocity_bins+duration_bins-1)
    Following every time_shift: prev -> prev + note_bins+velocity_bins+duration_bins
    """
    shift_v = note_bins
    shift_nd = note_bins + velocity_bins
    shift_d = note_bins + velocity_bins + duration_bins
    buffer_sz = (notes != 0).sum()
    buffer_sz = buffer_sz * 3 + notes.shape[0] - buffer_sz
    buffer = np.empty(buffer_sz, dtype=notes.dtype)
    b_idx = 0
    i = 0
    while i < notes.shape[0]:
        if notes[i] == 0:
            # a time-shift
            buffer[b_idx] = shift_d + durations[i]
            b_idx += 1
        else:
            # a note
            buffer[b_idx] = notes[i]
            buffer[b_idx + 1] = shift_v + velocities[i]
            buffer[b_idx + 2] = shift_nd + durations[i]
            b_idx += 3
        i += 1
    return buffer

def flatten_notes_(notes, velocities, durations, note_bins=89, velocity_bins=127, duration_bins=200):
    """
    Flattens bins such that:
    Following every note: (note: 0 -> note_bins-1), (velocity: prev -> prev + note_bins+velocity_bins-1), (duration_n_bin: prev -> prev + note_bins+velocity_bins+duration_bins-1)
    Following every time_shift: prev -> prev + note_bins+velocity_bins+duration_bins
    """
    shift_v = note_bins
    shift_nd = note_bins + velocity_bins
    shift_d = note_bins + velocity_bins + duration_bins
    
    buffer = []
    for i in range(notes.shape[0]):
        if notes[i] == 0:
            # a time-shift
            buffer.append(shift_d + durations[i])
        else:
            # a note
            buffer.append(notes[i])
            buffer.append(shift_v + velocities[i])
            buffer.append(shift_nd + durations[i])

    return np.array(buffer)

def unflatten_notes(flattened, note_bins=89, velocity_bins=127, duration_bins=200):
    """
    Inverse of flatten_notes
    """
    shift_v = note_bins
    shift_nd = note_bins + velocity_bins
    shift_d = note_bins + velocity_bins + duration_bins
    buffer_sz = (flattened < note_bins).sum()
    buffer_sz = buffer_sz + flattened.shape[0] - buffer_sz * 3
    buffer = np.empty((3, buffer_sz), dtype=flattened.dtype)

    b_idx = 0
    i = 0
    while b_idx < buffer_sz:
        if flattened[i] < note_bins:
            # following is a note
            buffer[0, b_idx] = np.clip(flattened[i], 0, note_bins-1)
            buffer[1, b_idx] = np.clip(flattened[i+1] - shift_v,  0, velocity_bins-1)
            buffer[2, b_idx] = np.clip(flattened[i+2] - shift_nd, 0, duration_bins-1)
            i += 3
        elif flattened[i] >= shift_d:
            buffer[0, b_idx] = 0
            buffer[1, b_idx] = 0
            buffer[2, b_idx] = np.clip(flattened[i] - shift_d, 0, duration_bins-1)
            i += 1
        else:
            i += 1
        b_idx += 1
    assert i == flattened.shape[0] - 1, ("actual: ", i, flattened.shape)
    return buffer 

def unflatten_notes_(flattened, note_bins=89, velocity_bins=127, duration_bins=200):
    """
    Inverse of flatten_notes
    """
    shift_v = note_bins
    shift_nd = note_bins + velocity_bins
    shift_d = note_bins + velocity_bins + duration_bins

    buffer_n = []
    buffer_v = []
    buffer_d = []
    # cut off head, since sequence may be randomly truncated
    lo = 0
    while not (flattened[lo] < note_bins):
        lo += 1

    for i in range(lo, flattened.shape[0]):
        if flattened[i] < note_bins:
            # following is a note
            buffer_n.append(np.clip(flattened[i], 0, note_bins-1))
        elif shift_v <= flattened[i] < shift_nd:
            buffer_v.append(np.clip(flattened[i] - shift_v,  0, velocity_bins-1))
        elif shift_nd <= flattened[i] < shift_d:
            buffer_d.append(np.clip(flattened[i] - shift_nd, 0, duration_bins-1))
        else:
            buffer_n.append(0)
            buffer_v.append(0)
            buffer_d.append(np.clip(flattened[i] - shift_d, 0, duration_bins-1))
    
    # cut off any unfinished notes, since sequence might be randomly truncated
    trunc = min(len(buffer_n), len(buffer_v), len(buffer_d))
    buffer_n = buffer_n[:trunc]
    buffer_v = buffer_v[:trunc]
    buffer_d = buffer_d[:trunc]
    return np.asarray([buffer_n, buffer_v, buffer_d])


@ray.remote
@dataclass
class UnifiedEncoding:
    """
    """
    seq_len: int = 1024
    notes: Tuple[int, int] = (-5, 5)
    velocities: Tuple[int, int] = (-7, 7)
    durations: Tuple[float, float] = (0.8, 1.2)
    duration_bins: int = 200
    note_bins: int = 89
    velocity_bins: int = 127
    clip_time_skip: int = 0

    def __post_init__(self):
        self.bin_arr = np.cumsum(pm.compute_lin_bin(0, 5, self.duration_bins))
        self.n_bins = self.note_bins + self.velocity_bins + self.duration_bins * 2 + 2

    def pad(self, arr: np.ndarray):
        arr += 2
        pad = self.seq_len - arr.shape[0] - 2
        return np.concatenate([[0], arr, np.ones(1 + pad, dtype=arr.dtype)])
      
    def transform_single_batch(self, notes: np.ndarray, velocities: np.ndarray, durations: np.ndarray) -> np.ndarray:
        assert notes.ndim == 1
        notes = copy.deepcopy(notes)
        velocities = copy.deepcopy(velocities)
        durations = copy.deepcopy(durations)

        notes[notes != 0] += np.random.randint(self.notes[0], self.notes[1])
        notes = np.clip(notes, 0, self.note_bins-1)

        velocities[velocities != 0] += np.random.randint(self.velocities[0], self.velocities[1])
        velocities = np.clip(velocities, 0, self.velocity_bins-1)

        durations = durations * np.random.uniform(self.durations[0], self.durations[1])
        durations = np.digitize(durations, self.bin_arr)
        durations = np.clip(durations, 0, self.duration_bins-1)

        data = flatten_notes(notes, velocities, durations, note_bins=self.note_bins, velocity_bins=self.velocity_bins, duration_bins=self.duration_bins)

        # truncate to required size
        if data.shape[0] > self.seq_len - 2:
            idx = np.random.randint(0, data.shape[0] - self.seq_len + 2)
            end = idx + self.seq_len - 2
            data = data[idx:end]
        
        data = self.pad(data)
        return data
    
    def filter_durations(self, stream: pm.MidiStream):
        data = np.transpose(stream.events, (1, 0))
        is_time_skip = data[0] == -1
        is_small = data[2] < self.clip_time_skip
        predicate = np.invert(is_time_skip & is_small)
        data = data[:, predicate]
        notes = data[0] + 1
        durations = data[2] / stream.ticks

        return notes, data[1], durations
    
    def call(self, data: List[pm.MidiStream]):
        data = map(self.filter_durations, data)
        data = np.stack([self.transform_single_batch(*d) for d in data], axis=0)
        return data[:, :-1], data[:, 1:]


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
    clip_time_skip: int = 0

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
            idx = np.random.randint(0, notes.shape[0] - self.seq_len + 2)
            end = idx + self.seq_len - 2
            notes = notes[idx:end]
            velocities = velocities[idx:end]
            durations = durations[idx:end]
        
        notes[notes != 0] += np.random.randint(self.notes[0], self.notes[1])
        notes = np.clip(notes, 0, self.note_bins)
        notes = self.pad(notes)

        velocities[velocities != 0] += np.random.randint(self.velocities[0], self.velocities[1])
        velocities = np.clip(velocities, 0, self.velocity_bins)
        velocities = self.pad(velocities)

        durations = durations * np.random.uniform(self.durations[0], self.durations[1])
        durations = np.digitize(durations, self.bin_arr)
        durations = np.clip(durations, 0, self.duration_bins)
        durations = self.pad(durations)

        return np.stack([notes, velocities, durations], axis=0)
    
    def filter_durations(self, stream: pm.MidiStream):
        data = np.transpose(stream.events, (1, 0))
        is_time_skip = data[0] == -1
        is_small = data[2] < self.clip_time_skip
        predicate = np.invert(is_time_skip & is_small)
        data = data[:, predicate]
        notes = data[0] + 1
        durations = data[2] / stream.ticks

        return notes, data[1], durations
    
    def call(self, data: List[pm.MidiStream]):
        data = map(self.filter_durations, data)
        data = np.stack([self.transform_single_batch(*d) for d in data], axis=0)
        return data[:, :, :-1], data[:, :, 1:]


@dataclass
class SeparatedReconstruct:
    """
    Reconstruct the midi stream from the outputs of SeparatedEncoding
    """
    ticks: int = 480
    duration_lin_bins: int = 200
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

        arr -= 2 # unshift padding from transfroms 
        arr[0] = np.clip(arr[0], 0, self.note_bins)
        arr[1] = np.clip(arr[1], 0, self.velocity_bins)
        arr[2] = np.clip(arr[2], 0, self.duration_lin_bins - 1)

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


@dataclass
class UnifiedReconstruct:
    """
    Reconstruct the midi stream from the outputs of SeparatedEncoding
    """
    ticks: int = 480
    duration_lin_bins: int = 200
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
            arr = arr[1:]
        if (arr == 1).any():
            ind = np.argmax(arr[0] == 1)
            arr = arr[:ind]

        arr -= 2 # unshift padding from transfroms
        arr = unflatten_notes_(arr)

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


def test_unified_encoding():
    from matplotlib import pyplot as plt

    ec = UnifiedEncoding.remote(seq_len=3000)
    file = r'datasets/maestro/maestro-v3.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi'
    raw_stream = pm.parse_raw_midi_stream(file)
    stream = pm.raw_stream_to_stream(raw_stream)
    result= ec.call.remote([stream])
    result = ray.get(result)[0][0]
    print(result)
    print(result.shape)
    plt.hist(result, bins=618)
    plt.show()
    print(result.max(), result.min())
    print(result.shape)
    rc = UnifiedReconstruct()
    rc.binned_encoding_to_file('reconstructed.midi', result)
    import shutil
    shutil.copy(file, 'original.midi')


if __name__ == "__main__":
    # test reconstruction
    test_unified_encoding()

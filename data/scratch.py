# %%
import time
from pathlib import Path
import ray
import numpy as np
from matplotlib import pyplot as plt
import shutil

import data.parse_midi_numba as pm

ray.init()
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


data_dir = r'/media/allan/DATA1/Productivity/Programs/datasets/Music_dataset/maestro-v3.0.0'
files = get_midi_files(data_dir)


# %%
cur_time = time.time()

streams = []
for i in range(len(files)):
    a = file_to_midi_norm_stream.remote(files[i])
    streams.append(a)
streams = ray.get(streams)

print(time.time() - cur_time)

# %%
durations = map(lambda x: x.events[:, 2] / x.ticks, streams)
durations = np.concatenate(list(durations))

velocities = map(lambda x: x.events[:, 1], streams)
velocities = np.concatenate(list(velocities))

notes = map(lambda x: x.events[:, 0], streams)
notes = np.concatenate(list(notes))

print("duration stats")
print(durations.max(), durations.min())
print(durations.dtype)
plt.hist(durations, bins=100)

print("velocity stats")
print(velocities.max(), velocities.min())
print(velocities.dtype)
print(velocities.shape)
plt.hist(velocities)

# %%
time_shifts = (notes == -1) & (durations < 20)
print(time_shifts.mean())

# %%
def compute_percentile_cutoff(durations, percent=0.99, bins=1000):
    counts, bins = np.histogram(durations, bins=bins)
    
    total = np.cumsum(counts, -1)
    perc  = total / total[-1]
    ind = np.argmax(perc > percent)
    # percent (99%) of total is under value
    value = bins[ind]
    return value


val = compute_percentile_cutoff(durations, percent=0.99, bins=100)
print('99 percent of duration is under', val)

# %%
def compute_bin_arr_exp(durations, exp_bins):
    lo = durations.min()
    hi = durations.max()
    exp_coef = (hi - lo) / (np.exp2(exp_bins) - 1)
    exp_arr = exp_coef * np.exp2(np.arange(1, exp_bins + 1, 1), dtype=np.float64)
    return exp_arr

def compute_lin_bin(lo, hi, bins):
    return np.full([bins], (hi - lo) / bins, dtype=np.float64)

def compute_bin_arr_piecewise2(durations, lin_bins, exp_bins):
    lo = durations.min()
    mi0 = compute_percentile_cutoff(durations, 0.3, bins=4000)
    mi1 = compute_percentile_cutoff(durations, 0.7, bins=3000)
    mi2 = compute_percentile_cutoff(durations, 0.99, bins=1000)
    hi = durations.max()
    print('lo:', lo)
    print('mi0:', mi0)
    print('mi1:', mi1)
    print('mi2:', mi2)
    print('hi:', hi)
    

    lin_arr0 = compute_lin_bin(lo, mi0, int(lin_bins * 0.2))
    lin_bins -= lin_arr0.shape[0]
    lin_arr1 = compute_lin_bin(lo, mi1, int(lin_bins * 0.4))
    lin_bins -= lin_arr1.shape[0]
    lin_arr2 = compute_lin_bin(mi1, mi2, lin_bins)

    exp_coef = (hi - mi2) / (np.exp2(exp_bins + 1) - 2)
    exp_arr = exp_coef * np.exp2(np.arange(1, exp_bins + 1, 1), dtype=np.float64)
    return np.cumsum(np.concatenate([lin_arr0, lin_arr1, lin_arr2, exp_arr]))

def compute_dynamic_bins(data, resolution_bins, bins):
    """assume that the histogram of data is generally decreasing"""
    bin_arr = np.zeros(bins, dtype=np.float64)
    counts, bounds = np.histogram(data, bins=resolution_bins)
    percents = counts / counts.sum()
    delta = bounds[1] - bounds[0]
    hi_bound = bounds[-1] + delta
    idx = bins
    running_percent = 0
    for i in range(resolution_bins - 1, -1, -1):
        running_percent += percents[i]
        bin_amount = int(running_percent * bins)
        if bin_amount > 1:
            bin_arr[idx-bin_amount:idx] = compute_lin_bin(bounds[i], hi_bound, bins=bin_amount)
            running_percent = 0
            hi_bound = bounds[i]
            idx -= bin_amount
    print(idx)
    if idx > 0:
        bin_arr[0:idx] = compute_lin_bin(data.min(), hi_bound, idx)
    print(bin_arr)
    return np.cumsum(bin_arr)

def compute_clippy_bin(durations, lin_bins, hard_clip):
    lo = 0
    mi0 = compute_percentile_cutoff(durations, 0.2, bins=4000)
    mi1 = compute_percentile_cutoff(durations, 0.8, bins=3000)
    mi2 = compute_percentile_cutoff(durations, 0.99, bins=1000)
    hi = hard_clip
    print('lo:', lo)
    print('mi0:', mi0)
    print('mi1:', mi1)
    print('mi2:', mi2)
    print('hi:', hi)
    

    lin_arr0 = compute_lin_bin(lo, mi0, int(lin_bins * 0.1))
    lin_bins -= lin_arr0.shape[0]
    lin_arr1 = compute_lin_bin(lo, mi1, int(lin_bins * 0.6))
    lin_bins -= lin_arr1.shape[0]
    lin_arr2 = compute_lin_bin(mi1, mi2, lin_bins)
    return np.cumsum(np.concatenate([lin_arr0, lin_arr1, lin_arr2]))

# lol, I spent all this time, but it turns out that linear bins are the best
bin_arr = np.cumsum(compute_lin_bin(0, 5, 200))
print(bin_arr.shape)
plt.plot(bin_arr)
plt.show()

# %%
x = np.linspace(0, 150, num=10000)

discretized = np.digitize(x, bin_arr, right=False)
plt.plot(x, discretized)
plt.show()

# %%
import copy
def reconstruction_error(arr1, arr2, lo, hi):
    """computes the l1 distance between arr1 and arr2 for values between lo and hi"""
    arr1 = copy.deepcopy(arr1)
    mask = np.invert((arr1 >= lo) & (arr1 < hi)) | np.invert((arr2 >= lo) & (arr2 < hi))
    arr1[np.where(mask)] = 0
    arr2 = copy.deepcopy(arr2)
    arr2[np.where(mask)] = 0
    return np.sum(np.abs(arr1 - arr2)) / np.sum(mask) * 100


def test_reconstruction(file, bin_arr):
    raw_stream = pm.parse_raw_midi_stream(file)
    stream = pm.raw_stream_to_stream(raw_stream)

    time_shifts = (stream.events[:, 0] == -1) & (stream.events[:, 2] < 20)
    print(time_shifts.mean())
    keep_shifts = np.invert(time_shifts)
    stream.events = stream.events[keep_shifts, :]
    
    original_durations = copy.deepcopy(stream.events[:, 2])
    durations = stream.events[:, 2] / stream.ticks
    durations = np.clip(durations, 0, 7)
    durations = np.digitize(durations, bin_arr, right=False)
    durations[np.where(durations == bin_arr.shape[0])] -= 1
    durations = bin_arr[durations]
    stream.events[:, 2] = (durations * stream.ticks).astype(stream.events.dtype)
    new_durations = stream.events[:, 2]

    print(reconstruction_error(original_durations, new_durations, 1, 30))
    print(reconstruction_error(original_durations, new_durations, 30, 100))
    print(reconstruction_error(original_durations, new_durations, 101, 300))
    print(reconstruction_error(original_durations, new_durations, 300, 700))

    raw_stream1 = pm.stream_to_raw_stream(stream)
    pm.raw_stream_to_midi_file('reconstructed.midi', raw_stream1)
    shutil.copy(file, 'original.midi')

test_reconstruction(files[1], bin_arr)

# TODO: Make data-dependent arr-bin function
# %%
import numpy as np
from matplotlib import pyplot as plt
from misc import EasyDict
import ray

ray.init()

from data.datasetv2 import Dataset
from data.transforms import SeparatedEncoding, SeparatedReconstruct

d_args = EasyDict()

data_dir = r'datasets/maestro/maestro-v3.0.0'
d_args.batch_size            = 4
d_args.workers               = 2
d_args.seq_len               = 1025               # one extra token since loss is auto-regressive
d_args.duration_lin_bins     = 200                # number of linear (same width) bins/embeddings for duration
d_args.note_shifts           = (-5, 5)            # random integer shifts in note for augmentation
d_args.velocity_shifts       = (-7, 7)            # augment velocities as well
d_args.duration_muls         = (0.8, 1.2)         # random multiplication to duration from uniform distribution


# %%
actors = [SeparatedEncoding.remote(
    d_args.seq_len,
    d_args.note_shifts,
    d_args.velocity_shifts,
    d_args.duration_muls,
    d_args.duration_lin_bins,
    clip_time_skip=15
) for _ in range(d_args.workers)]
dataset = Dataset(data_dir, d_args.batch_size, actors)

# %%
x, y = dataset.get()
plt.hist(x[0, 0], bins=92)

# %%
print((x[:, 0] == 2).sum() / np.prod(x[:, 0].shape))

# %%
rc = SeparatedReconstruct(duration_lin_bins=200)
rc.binned_encoding_to_file("reconstructed.midi", dataset.get()[0][0])

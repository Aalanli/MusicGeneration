# %%
import torch
import numpy as np
from tqdm import tqdm
from data.datasetv2 import Reconstruct
from train_utils import load_checkpoint
from model_baseline.build_everything import model, dataset


def dummy_model(x: torch.Tensor):
    b, _, n = x.shape
    mk = lambda l: torch.zeros(b, n, l, dtype=x.dtype, device=x.device)
    return mk(25), mk(63), mk(166)


def inference(self, seq: torch.Tensor, idx=1, deterministic=False):
    with torch.no_grad():            
        for i in tqdm(range(idx, seq.shape[-1] - 1)):
            n, v, d = self(seq)
            if not deterministic:
                r = map(lambda x: torch.multinomial(x[:, i + 1].softmax(-1), num_samples=1), [n, v, d])
            else:
                r = map(lambda x: torch.argmax(x[:, i + 1], -1), [n, v, d])
            r = torch.stack(list(r), -1)
            seq[:, :, i] = r
    return seq


run_name = 'baseline_1.1'
run_dir  = f'experiments/{run_name}'

load_checkpoint(run_dir, model)
model = model.cuda()

# %%
x, y = dataset.get()
x = x[0]
x = torch.from_numpy(x).cuda()
y = torch.from_numpy(y).cuda()

yn, yv, yd = model(x)
print((y[:, 0] == yn.argmax(-1)).float().mean())

# %%
seq = torch.zeros([1, 3, 512], dtype=torch.int32).cuda()
_, primer = dataset.get()

seq[:, :, 1:64] = torch.from_numpy(primer[0:1, :, :63]).cuda()
print(seq)

seq = inference(model, seq, deterministic=False)
seq[:, :, -1] = 1
seq = seq[0].cpu().numpy()
print(seq)
rec = Reconstruct(duration_lin_bins=200)
rec.binned_encoding_to_file('gen_test1.midi', seq)

# %%
rec = Reconstruct(duration_lin_bins=200)
rec.binned_encoding_to_file("get_test1.midi", dataset.get()[1][0])
# %%
from matplotlib import pyplot as plt
plt.hist(seq[2], bins=100)


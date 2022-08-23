# %%
import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from data.transforms import UnifiedReconstruct
from train_utils import load_checkpoint
from model_baseline.build_everything_unified import model, dataset


def inference(self, seq: torch.Tensor, idx=1, deterministic=False):
    with torch.no_grad():            
        for i in tqdm(range(idx, seq.shape[-1] - 1)):
            n = self(seq)
            if not deterministic:
                r = torch.multinomial(n[:, i + 1].softmax(-1), num_samples=1)
            else:
                r = torch.argmax(n[:, i + 1], -1)
            seq[:, i] = r
    return seq

run_name = 'baseline_3.1'
run_dir  = f'experiments/{run_name}'

load_checkpoint(run_dir, model)
model = model.cuda()

# %%
x, y = dataset.get()
plt.hist(x[0], bins=618)
plt.show()

print(x.max())
print(np.histogram(x[0, 0], bins=618, range=(0, 618)))
x = torch.from_numpy(x).cuda()
y = torch.from_numpy(y).cuda()

yn = model(x)
print((y == yn.argmax(-1)).float().mean())


# %%
seq = torch.zeros([1, 1023], dtype=torch.int32).cuda()
_, primer = dataset.get()

seq[:, 1:64] = torch.from_numpy(primer[0:1, :63]).cuda()
print(seq)

seq = inference(model, seq, deterministic=False)
seq[:, -1] = 1
seq = seq[0].cpu().numpy()
print(seq)
rec = UnifiedReconstruct()
rec.binned_encoding_to_file('gen_test1.midi', seq)

# %%
rec.binned_encoding_to_file("get_test1.midi", dataset.get()[0][0])
# %%
plt.hist(seq[1], bins=100)


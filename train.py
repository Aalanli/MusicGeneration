# %%
import torch

from torch.utils.data import DataLoader
from data.dataset import Data

from alibi import AlibiTransformer
from fixed_pos import RotaryTransformer
from relative_pos import RelativeTransformer
from sparse_alibi import AlibiTransformer as SparseTransformer
from model_utils import TrainerWandb, calculate_perplexities_pure, calculate_perplexities_teacher
from misc import EasyDict, CustomScheduler

data_dir = r'/media/allan/DATA1/Productivity/Programs/datasets/Music_dataset/maestro-v3.0.0'

args = EasyDict()

args.n_vocab = 246
args.embed_dim = 512
args.d_model = 512
args.proj_forward = 2048
args.n_layers = 4
args.n_heads = 8
args.dropout = 0.1
args.max_sequence = 768
args.sparse_dim = 32

t_args = EasyDict()
t_args.batch_size = 64
t_args.batch_splits = 8
t_args.warmup_steps = 10000
t_args.name = "sparse_formerV12"
t_args.sparse_loss_coef = 0.3


#perf_dataset = train_data = Data('/home/allan/Programs/Music_generation/data/datasets/torch/train1', 1024)
#perf_dataset = DataLoader(perf_dataset, 8, shuffle=True, num_workers=8)

train_data = Data('/home/allan/Programs/Music_generation/data/datasets/torch/train1', args.max_sequence + 1)
test_data = Data('/home/allan/Programs/Music_generation/data/datasets/torch/test', args.max_sequence + 1)

train_data = DataLoader(train_data, t_args.batch_size, shuffle=True, num_workers=8)
test_data = DataLoader(test_data, 2, shuffle=True, num_workers=4)


from matplotlib import pyplot as plt


model = SparseTransformer(**args).cuda()

args.update(t_args)

optimizer = torch.optim.Adam(model.parameters())
scheduler = CustomScheduler(optimizer, args.d_model, t_args.warmup_steps)
trainer = TrainerWandb(model, 
                        optimizer, 
                        'experiments', 
                        metric_step=50, 
                        checkpoint_step=200, 
                        mixed_precision=False, 
                        lr_scheduler=scheduler, 
                        max_checkpoints=10, 
                        config=args)
# %%
x = next(iter(train_data)).cuda()
x = x[:4, :-1]
print(x.shape)
y,_ = model(x)
print(torch.isnan(y).any())

# %%
trainer.train_epochs(epochs=1, train_data=train_data, eval_data=test_data)
trainer.reset_ab(1, 1)
trainer.train_epochs(epochs=1, train_data=train_data, eval_data=test_data)
trainer.reset_ab(1, 2.0)
trainer.train_epochs(epochs=1, train_data=train_data, eval_data=test_data)

"""    
    with torch.cuda.amp.autocast():
        perplexity = calculate_perplexities_teacher(model, next(iter(perf_dataset)).cuda(), sample_sizes=[1024 + i * 64 for i in range(1, 3)], window=None)
        print(perplexity)
        plt.plot([i[0] for i in perplexity], [i[1] for i in perplexity], label=args.name)

plt.legend()
plt.show()
"""
# %%
x = next(iter(train_data))[0:4]
x = x.cuda()
y, sp = model(x[:, :-1], get_gate=True)

# %%
with torch.no_grad():
    for (w, m) in sp:
        print(w)
        print(m.max(), m.min())
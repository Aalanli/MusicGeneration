# %%
import torch

from torch.utils.data import DataLoader
from data.dataset import Data

from alibi import AlibiTransformer
from fixed_pos import RotaryTransformer
from relative_pos import RelativeTransformer
from model_utils import TrainerWandb, calculate_perplexities_pure, calculate_perplexities_teacher
from misc import EasyDict, CustomScheduler


args = EasyDict()

args.n_vocab = 246
args.d_model = 512
args.proj_forward = 1024
args.n_layers = 6
args.n_heads = 8
args.dropout = 0.1
args.max_seq_len = 1024

args.batch_size = 64
args.batch_splits = 8

args.warmup_steps = 10000

perf_dataset = train_data = Data('/home/allan/Programs/Music_generation/data/datasets/torch/train1', 4096)
perf_dataset = DataLoader(perf_dataset, 8, shuffle=True, num_workers=8)

train_data = Data('/home/allan/Programs/Music_generation/data/datasets/torch/train1', args.max_seq_len + 1)
test_data = Data('/home/allan/Programs/Music_generation/data/datasets/torch/test', args.max_seq_len + 1)

train_data = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=8)
test_data = DataLoader(test_data, 2, shuffle=True, num_workers=4)

alibi_args = args.get_copy()
alibi_args.name = 'alibiTransformer'

rotary_args = args.get_copy()
rotary_args.name = 'rotaryTransformer'

relative_args = args.get_copy()
relative_args.name = 'relativeTransformer'


train = [(rotary_args, RotaryTransformer), (alibi_args, AlibiTransformer), (relative_args, RelativeTransformer)]


from matplotlib import pyplot as plt
        
for args, model in train[1:]:
    model = model(args.n_vocab, args.d_model, args.n_layers, args.n_heads, args.max_seq_len, args.proj_forward, args.dropout).cuda()
    
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = CustomScheduler(optimizer, args.d_model, args.warmup_steps)
    trainer = TrainerWandb(model, 
                           optimizer, 
                           'experiments', 
                           metric_step=50, 
                           checkpoint_step=200, 
                           mixed_precision=True, 
                           lr_scheduler=scheduler, 
                           max_checkpoints=10, 
                           config=args)
    trainer.train_epochs(epochs=2, train_data=train_data, eval_data=test_data)

"""    
    with torch.cuda.amp.autocast():
        perplexity = calculate_perplexities_teacher(model, next(iter(perf_dataset)).cuda(), sample_sizes=[1024 + i * 64 for i in range(1, 3)], window=None)
        print(perplexity)
        plt.plot([i[0] for i in perplexity], [i[1] for i in perplexity], label=args.name)

plt.legend()
plt.show()
"""


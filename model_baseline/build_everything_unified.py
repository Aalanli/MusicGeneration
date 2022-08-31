# %%
from setup import setup_env
# installs any packages needed for the computation below to work
setup_env()


import torch
import wandb
import ray
from misc import EasyDict


###############################################################
# build model first, since optimizers and lr will need it later
###############################################################
from model_baseline.alibi_unified import build_model_and_criterion, Metrics

m_args = EasyDict()

m_args.vocab                 = 89 + 127 + 200 * 2 + 2     # note_bins + velocity_bins + duration_n_bins + duration_shift_bins + bos + eos
m_args.embed_dim             = 1024
m_args.n_layers              = 6
m_args.n_heads               = 8
m_args.max_sequence          = 1024
m_args.proj_forward          = 4096
m_args.dropout               = 0.1 
m_args.activation            = 'gelu'
m_args.pad_index             = 1                      # also the eos token


model, criterion = build_model_and_criterion(m_args)
train_metrics = Metrics()
eval_metrics = Metrics()

##################################
# simple optimizer hyperparameters
##################################
opt_args = EasyDict()
opt_args.lr                = 1e-3
opt_args.weight_decay      = 1e-4

optimizer = torch.optim.AdamW(model.parameters(), **opt_args)

opt_args.lr_drop           = 50
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt_args.lr_drop)


################################################
# build the dataset and data transformations
# since Dataset is generic w.r.t the maestro 
# dataset, but transforms may be model dependent
################################################

from data.datasetv2 import Dataset, get_midi_files
from data.transforms import UnifiedEncoding

d_args = EasyDict()

data_dir = r'datasets/maestro/maestro-v3.0.0'
d_args.batch_size            = 8
d_args.workers               = 2
d_args.seq_len               = 1025               # one extra token since loss is auto-regressive
d_args.duration_bins         = 200                # number of linear (same width) bins/embeddings for duration
d_args.note_shifts           = (-5, 5)            # random integer shifts in note for augmentation
d_args.velocity_shifts       = (-7, 7)            # augment velocities as well
d_args.duration_muls         = (0.8, 1.2)         # random multiplication to duration from uniform distribution
d_args.clip_time_skip        = 10                 # most time skips are too small to matter, removes any under this value
d_args.train_split           = 0.9
d_args.eval_batch_size       = 2

files = get_midi_files(data_dir)
ntrain_files = int(len(files) * d_args.train_split)
train_files = files[:ntrain_files]
eval_files = files[ntrain_files:]
print("train files:", len(train_files))
print("eval_files:", len(eval_files))

actors = [UnifiedEncoding.remote(
    d_args.seq_len,
    d_args.note_shifts,
    d_args.velocity_shifts,
    d_args.duration_muls,
    d_args.duration_bins,
    clip_time_skip=d_args.clip_time_skip
) for _ in range(d_args.workers)]
train_dataset = Dataset(data_dir, d_args.batch_size, actors, files=train_files)
eval_dataset = Dataset(data_dir, d_args.eval_batch_size, actors, files=eval_files)


# %%
###################
# commence training
###################
if __name__ == '__main__':
    import os
    from train_utils import training_loop, Logger
    run_name = 'baseline_3.5'
    train_args = EasyDict()
    train_args.epochs                     = 120
    train_args.run_dir                    = f'experiments/{run_name}'
    train_args.log_scalar_metric_step     = 100 * d_args.batch_size
    train_args.checkpoint_step            = 500 * d_args.batch_size
    train_args.batch_size                 = d_args.batch_size
    train_args.mixed_precision            = False


    configs = EasyDict()
    configs.model_args                    = m_args
    configs.optimizer_args                = opt_args
    configs.data_args                     = d_args
    configs.batch_size                    = d_args.batch_size
    configs.description                   = 'weight different classes differently'

    if not os.path.exists(train_args.run_dir):
        os.makedirs(train_args.run_dir)
    
    run_fn = lambda: wandb.init(project='MusicGeneration', entity='allanl', dir=train_args.run_dir, 
        group=run_name, id=run_name, config=configs, resume='allow')
    train_logger = Logger.remote(run_fn, train_metrics)
    eval_logger = Logger.remote(run_fn, eval_metrics)

    model = model.cuda()
    criterion = criterion.cuda()

    run = run_fn()
    with run:
        wandb.watch(model, log_freq=500)
        training_loop(model, train_dataset, criterion, optimizer, lr_scheduler, train_logger, eval_logger=eval_logger, eval_set=eval_dataset, **train_args)

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
from model_baseline.alibi_v2 import build_model_and_criterion, Metrics

m_args = EasyDict()

m_args.vocab_note            = 89  + 3     # extra padding tokens
m_args.vocab_velocity        = 127 + 3     # extra padding tokens
m_args.vocab_duration        = 200 + 3     # extra padding tokens
m_args.embed_note            = 128 * 2
m_args.embed_velocity        = 128 * 2
m_args.embed_duration        = 256 * 2
m_args.n_layers              = 10
m_args.n_heads               = 8
m_args.max_sequence          = 1024
m_args.proj_forward          = 4096
m_args.dropout               = 0.1 
m_args.activation            = 'gelu'
m_args.loss_weights          = {'loss_notes': 1, 'loss_velocity': 1, 'loss_duration': 1}


model, criterion = build_model_and_criterion(m_args)
metrics = Metrics()

##################################
# simple optimizer hyperparameters
##################################
opt_args = EasyDict()
opt_args.lr                = 1e-4
opt_args.weight_decay      = 1e-4

optimizer = torch.optim.AdamW(model.parameters(), **opt_args)

opt_args.lr_drop           = 50
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt_args.lr_drop)


################################################
# build the dataset and data transformations
# since Dataset is generic w.r.t the maestro 
# dataset, but transforms may be model dependent
################################################
ray.init()

from data.datasetv2 import Dataset
from data.transforms import SeparatedEncoding

d_args = EasyDict()

data_dir = r'datasets/maestro/maestro-v3.0.0'
d_args.batch_size            = 8
d_args.workers               = 2
d_args.seq_len               = 1025               # one extra token since loss is auto-regressive
d_args.duration_lin_bins     = 200                # number of linear (same width) bins/embeddings for duration
d_args.note_shifts           = (-5, 5)            # random integer shifts in note for augmentation
d_args.velocity_shifts       = (-7, 7)            # augment velocities as well
d_args.duration_muls         = (0.8, 1.2)         # random multiplication to duration from uniform distribution
d_args.clip_time_skip        = 10                 # most time skips are too small to matter, removes any under this value

actors = [SeparatedEncoding.remote(
    d_args.seq_len,
    d_args.note_shifts,
    d_args.velocity_shifts,
    d_args.duration_muls,
    d_args.duration_lin_bins,
    clip_time_skip=d_args.clip_time_skip
) for _ in range(d_args.workers)]
dataset = Dataset(data_dir, d_args.batch_size, actors)


# %%
###################
# commence training
###################
if __name__ == '__main__':
    from train_utils import training_loop, Logger
    run_name = 'baseline_2.91'
    train_args = EasyDict()
    train_args.epochs                     = 120
    train_args.run_dir                    = f'experiments/{run_name}'
    train_args.log_scalar_metric_step     = 100 * d_args.batch_size
    train_args.checkpoint_step            = 500 * d_args.batch_size
    train_args.batch_size                 = d_args.batch_size
    train_args.mixed_precision            = True


    configs = EasyDict()
    configs.model_args                    = m_args
    configs.optimizer_args                = opt_args
    configs.data_args                     = d_args
    configs.batch_size                    = d_args.batch_size
    configs.description                   = 'weight different classes differently'

    run_fn = lambda: wandb.init(project='MusicGeneration', entity='allanl', dir=train_args.run_dir, 
        group=run_name, id=run_name, config=configs, resume='allow')
    logger = Logger.remote(run_fn, metrics)

    model = model.cuda()
    criterion = criterion.cuda()

    run = run_fn()
    with run:
        wandb.watch(model, log_freq=500)
        training_loop(model, dataset, criterion, optimizer, lr_scheduler, logger, **train_args)

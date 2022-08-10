# %%
from setup import download_dataset, install_dependencies

download_dataset()
install_dependencies()

import torch
import wandb
import ray
from misc import EasyDict


###################################
from alibi_v2 import build_model_and_criterion, Metrics

m_args = EasyDict()

m_args.vocab_note            = 89  + 3
m_args.vocab_velocity        = 127 + 3
m_args.vocab_duration        = 200 + 3
m_args.embed_note            = 512
m_args.embed_velocity        = 256
m_args.embed_duration        = 256
m_args.n_layers              = 6
m_args.n_heads               = 8
m_args.max_sequence          = 1024
m_args.proj_forward          = 4096
m_args.dropout               = 0.1 
m_args.activation            = 'gelu'
m_args.eos_coef              = 0.5
m_args.eos_tokens            = [0, 1]
m_args.weights               = {'loss_notes': 1, 'loss_velocity': 1, 'loss_duration': 1}

model, criterion = build_model_and_criterion(m_args)
metrics = Metrics()

####################################
opt_args = EasyDict()
opt_args.lr                = 1e-4
opt_args.weight_decay      = 1e-4

optimizer = torch.optim.AdamW(model.parameters(), **opt_args)

opt_args.lr_drop           = 50
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt_args.lr_drop)

ray.init()

from data.datasetv2 import Dataset
d_args = EasyDict()


data_dir = r'/datasets/maestro/maestro-v3.0.0'
d_args.batch_size            = 4
d_args.workers               = 2
d_args.seq_len               = 1024
d_args.duration_lin_bins     = 200
d_args.note_shifts           = (-5, 5)
d_args.velocity_shifts       = (-7, 7)
d_args.duration_muls         = (0.8, 1.2)

dataset = Dataset(data_dir, **d_args)

# %%
####################################
if __name__ == '__main__':
    from train_utils import training_loop, Logger
    run_name = 'baseline_1.2'
    train_args = EasyDict()
    train_args.epochs                     = 10
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
    configs.description                   = 'naive baseline model with no adjustments'

    run_fn = lambda: wandb.init(project='MusicGeneration', entity='allanl', dir=train_args.run_dir, 
        id=run_name, config=configs, resume='allow')
    logger = Logger.remote(run_fn, metrics)

    model = model.cuda()
    criterion = criterion.cuda()

    train_args.epochs = 120
    training_loop(model, dataset, criterion, optimizer, lr_scheduler, logger, **train_args)

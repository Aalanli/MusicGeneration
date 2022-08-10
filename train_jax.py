# %%
import functools
import os
import math
from dataclasses import asdict
from types import MethodType

import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
from flax.training.checkpoints import save_checkpoint, latest_checkpoint, restore_checkpoint
from flax import linen as nn
import optax
import tqdm

import wandb


from jax_transformer import AlibiTransformer, construct_alibi
from data.data_jax import Data

def get_dataset(batch_size, seq_len):
    datadir = '/home/allan/Programs/Music_generation/data/datasets/torch/train1'
    data = Data(datadir, seq_len, batch_size)
    return data

def scheduler(d_model: int, warmup_steps: int) -> optax.Schedule:
    slope = warmup_steps ** (-1.5)
    const = 1 / math.sqrt(d_model)
    def lr_schedule1(step):
        arg1 = 1 / jnp.sqrt(step + 1)
        arg2 = (step + 1) * slope
        return const * jnp.minimum(arg1, arg2)
    return lr_schedule1

def get_ckpt_states(model: train_state.TrainState):
    param_dict = asdict(model)
    checkpointable = ['step', 'params', 'opt_state']
    return {k: v for (k, v) in param_dict.items() if k in checkpointable}

def get_model(rng, args):
    model = AlibiTransformer(args['heads'], args['vocab'], args['d_model'], args['n_layers'])
    params = model.init(rng, jnp.ones((4, 100), jnp.int32))
    tx = optax.adam(args['lr'])
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def log_scalars(log: dict, step, prefix=''):
    prefix = prefix + "/"
    prefix_copy = {f'{(prefix + k)}': v for k, v in log.items()}
    wandb.log(prefix_copy, step=step)

def zero_scalar(log: dict):
    for k in log: log[k] = 0

def div_scalar(log: dict, a):
    for k in log: log[k] = log[k] / a

def sum_scalar(log1: dict, log2: dict):
    for k in log1: log1[k] += log2[k]

@jax.jit
def cross_entropy(logits, truth):
    logits = jax.nn.log_softmax(logits, -1)
    truth = jnp.expand_dims(truth, -1)
    index_row = jax.vmap(lambda a, b: a[b], 0)
    index_batch = jax.vmap(index_row, 0)
    vals = index_batch(logits, truth)
    return -jnp.mean(vals) 

@functools.partial(jax.jit, static_argnums=2)
def train_step(model: train_state.TrainState, data, lr_scheduler=None):
    def loss_fn(params):
        logits = model.apply_fn(params, data[:, :-1])
        loss = cross_entropy(logits, data[:, 1:])
        loss /= data.shape[0] # scale loss by batch size, so every run with different batch size is consistent
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == data[:, 1:])
    model = model.apply_gradients(grads=grads)

    metric = {'loss': loss, 'accuracy': accuracy}
    if lr_scheduler is not None:
        metric['lr'] = lr_scheduler(model.step)
    return model, metric


def train_steps(steps, model: train_state.TrainState, data_gen, metric_step, check_point_step, root_dir, lr_scheduler=None):
    metrics = {'accuracy': 0, 'loss': 0}
    log_steps = model.step
    for i in tqdm.tqdm(range(steps)):
        data = data_gen.get()
        model, metric0 = train_step(model, data, lr_scheduler)
        sum_scalar(metrics, metric0)
        if ((log_steps + 1) % metric_step) == 0:
            div_scalar(metrics, metric_step)
            log_scalars(metrics, log_steps // 16, 'train')
            zero_scalar(metrics)
        log_steps += 1
        if ((log_steps + 1) % check_point_step) == 0:
            save_checkpoint(root_dir, get_ckpt_states(model), log_steps, keep=5, overwrite=True)
    return model

lr_schedule = scheduler(512, 50000)

args = dict(
    heads=8,
    vocab=248,
    d_model=512,
    n_layers=8,
    lr=lr_schedule,
    batch_size=4,
    train_seq_len=1025
)

rng = jax.random.PRNGKey(0)
model = get_model(rng, args)
data_gen = get_dataset(args['batch_size'], args['train_seq_len'])

model_name = 'jax-test12'
steps = 64 * 7000
checkpoint_step = 3000
metric_step = 500


model_dir = f'experiments/{model_name}'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
else:
    params = restore_checkpoint(model_dir, get_ckpt_states(model))
    model = model.replace(**params)
    print("restored from:", latest_checkpoint(model_dir))


print("params: ", jax.tree_util.tree_reduce(lambda v, ts: math.prod(ts.shape) + v, model.params, 0))

run = wandb.init(project='MusicGeneration', entity='allanl', dir=model_dir, id=model_name, resume='allow', config=args)
with run:
    model = train_steps(steps, model, data_gen, metric_step, checkpoint_step, model_dir, lr_scheduler=lr_schedule)


# %%
a = data_gen.get()[0:2]
print(a.shape)


# %%
@jax.jit
def forward_model(model, data, mask):
    return model.apply_fn(model.params, data, mask)

@jax.jit
def get_tokens(model, data):
    return jnp.argmax(forward_model(model, data, None), -1)

# %%

def calculate_true_accuracy(model, data):
    true = jnp.zeros([1])
    total = jnp.zeros([1])
    mask = construct_alibi(data.shape[-1], 8)
    for i in tqdm.tqdm(range(2, data.shape[-1] - 1)):
        pred = forward_model(model, data[:, 0:i], mask[:, :i, :i])
        true += (jnp.argmax(pred[0, -1], -1) == data[0, i+1]).astype(jnp.int32)
        total += 1

    return (true / total)

calculate_true_accuracy(model, a[:, :64])
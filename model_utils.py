import os
import pathlib
import dill
from typing import List, Tuple, Union
from itertools import cycle
from torch.nn.modules.module import T
from tqdm import tqdm

import torch
import wandb

from sparse_mask import compute_sparsity

class Trainer:
    """The general trainer base class"""
    def __init__(
        self, 
        model: torch.nn.Module, 
        criterion: torch.nn.Module = None, 
        optimizer: torch.optim.Optimizer = None, 
        directory: str = None, 
        metric_step: int = None, 
        checkpoint_step: int = None, 
        mixed_precision: bool = False, 
        lr_scheduler: bool = None,
        max_checkpoints: int = 5,
        config=None
    ) -> None:
    
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.mixed_precison = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
        self.dir = directory
        self.metric_step = metric_step
        self.checkpoint_step = checkpoint_step
        self.config = config

        self.max_checkpoints_saved = max_checkpoints
        self.steps = 0

        self.checkpointable = ['model', 'criterion', 'optimizer', 'scaler', 'steps']
        
        if isinstance(self.model, torch.nn.Module):
            if os.path.exists(self.dir):
                self.restore_latest_checkpoint()
            else:
                # first time loading model
                os.makedirs(self.dir)
                with open(self.dir + '/model_params.pkl', 'rw') as f:
                    dill.dump(config, f)
                # save a pickled referance of the objects
                # in case original class and arguments were forgotten
                with open(os.path.join(self.dir, 'obj_ref.pkl'), 'wb') as f:
                    dill.dump(self.obj_ref, f)
        elif isinstance(self.model, str):
            # if model is a string to the model directory
            self.restore_objects(os.path.join(self.model, 'obj_ref.pkl'))
            self.restore_latest_checkpoint()
    
    @property
    def obj_ref(self):
        return self.__dict__

    def restore_objects(self, file):
        with open(file, 'rb') as f:
            self.__dict__ = dill.load(f)

    def restore_checkpoint(self, file):
        checkpoint = torch.load(file)
        for k in checkpoint:
            if getattr(self.obj_ref[k], 'load_state_dict', None) is not None:
                self.obj_ref[k].load_state_dict(checkpoint[k])
            else:
                self.obj_ref[k] = checkpoint[k]
    
    def restore_latest_checkpoint(self):
        ckpt = self.get_checkpoints()
        if ckpt is not None and len(ckpt) != 0:
            self.restore_checkpoint(ckpt[-1])
            print('Restored checkpoint', ckpt[-1])
    
    def save_checkpoint(self, file):
        checkpoints = {}
        for k in self.checkpointable:
            if getattr(self.obj_ref[k], 'state_dict', None) is not None:
                checkpoints[k] = self.obj_ref[k].state_dict()
            else:
                checkpoints[k] = self.obj_ref[k]
        torch.save(checkpoints, file + '.tar')
    
    def get_checkpoints(self):
        x = sorted(pathlib.Path(self.dir).glob('**/*.tar'))
        if x is None:
            return None
        x = [str(i) for i in x]
        x = [int(os.path.basename(i)[:-4]) for i in x]  # sort by int
        x.sort()
        x = [self.dir + f'/{str(i)}.tar' for i in x]
        return x

    def regulate_checkpoints(self):
        ckpt = self.get_checkpoints()
        if ckpt is not None and len(ckpt) > self.max_checkpoints_saved:
            os.remove(ckpt[0])
        self.save_checkpoint(f'{self.dir}/{self.steps}')
    
    def save_model(self, path=None):
        if path is None:
            torch.save(self.model, f'{self.dir}/model')
        else:
            torch.save(self.model, path)
    
    def load_saved_model(self, path=None):
        """loads a model from the saved model file"""
        if path is None:
            self.model = torch.load(f'{self.dir}/model')
        else:
            self.model = torch.load(path)

    def sum_scalars(self, log: dict, new_log: dict):
        for k in log:
            log[k] += new_log[k]
    
    def div_scalars(self, log: dict, div: int):
        for k in log:
            log[k] /= div
    
    def zero_scalars(self, log: dict):
        for k in log:
            log[k] = 0
    
    def log_scalars(self, writer, log: dict, prefix=''):
        for k in log:
            writer.add_scalar(f'{prefix}/{k}', log[k] / self.metric_step, self.steps)


class TrainerWandb(Trainer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        directory: str = None,
        metric_step: int = None,
        checkpoint_step: int = None,
        mixed_precision: bool = None,
        lr_scheduler: bool = None,
        max_checkpoints: int = None,
        config=None,
        init_a=1.0,
        init_b=0.5) -> None:

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.mixed_precison = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)  # Ignore zero padding
        self.metric_step = metric_step
        self.checkpoint_step = checkpoint_step
        self.config = config

        self.max_checkpoints_saved = max_checkpoints
        self.steps = 0

        self.checkpointable = ['model', 'optimizer', 'scaler', 'steps']
        
        self.id = self.config.name
        self.dir = os.path.join(directory, self.id)
        self.batch_splits = self.config.batch_splits

        self.sparse_loss_coef = config.sparse_loss_coef

        if isinstance(self.model, torch.nn.Module):    
            if os.path.exists(self.dir):
                self.restore_latest_checkpoint()
            else:
                # first time loading model
                os.makedirs(self.dir)
                with open(self.dir + '/model_params.pkl', 'wb') as f:
                    dill.dump(config, f)
                # save a pickled referance of the objects
                # in case original class and arguments were forgotten
                with open(os.path.join(self.dir, 'obj_ref.pkl'), 'wb') as f:
                    dill.dump(self.obj_ref, f)
        elif isinstance(self.model, str):
            # if model is a string to the model directory
            self.restore_objects(os.path.join(self.model, 'obj_ref.pkl'))
            self.restore_latest_checkpoint()
        
        self.init_a = init_a
        self.init_b = init_b
    
    def reset_ab(self, a, b):
        self.init_a = a
        self.init_b = b
    
    def compute_sparsity(self, sample):
        with torch.no_grad():
            x = sample[0:4, :-1]
            _, sparsities = self.model(x, self.init_a, self.init_b, get_gate=True)
            masks = [i[1] for i in sparsities]
            sparsities = {i: compute_sparsity(mask) for i, mask in enumerate(masks)}
            return sparsities

    def train_step(self, sample):
        y = sample[:, 1:]
        x = sample[:, :-1]
        self.model.train()
        for x1, y1 in zip(x.chunk(self.batch_splits), y.chunk(self.batch_splits)):
            with torch.cuda.amp.autocast(enabled=self.mixed_precison):
                logits, sparsities = self.model(x1, self.init_a, self.init_b)  # logits = [batch, seq_len, classes]
                logits = logits.transpose(-1, -2)  # loss expects logit classes in dim 1
                pred_loss = self.loss(logits, y1).div(self.batch_splits)

                sp_lossl = [t for t in sparsities]
                sp_loss = self.sparse_loss_coef * sum(sp_lossl) / (len(sp_lossl) * self.batch_splits)
                loss = (pred_loss + sp_loss)
            
            self.scaler.scale(loss).backward()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(self.steps)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.steps += 1

        accuracy = accuracy_fn(y1, logits)
        sparsities = {f'sparsity_layer_{i}': float(x.sum()) for i, x in enumerate(sp_lossl)}
        metrics = {'loss': float(pred_loss), 'total_loss': float(loss), 'sparsity_loss': sp_loss, 'accuracy': float(accuracy)}
        metrics.update(sparsities)
        return metrics
    
    def eval_step(self, sample):
        with torch.cuda.amp.autocast(enabled=self.mixed_precison):
            with torch.no_grad():
                y = sample[:, 1:]
                x = sample[:, :-1]
                self.model.eval()

                logits, sparsities = self.model(x)
                logits = logits.transpose(-1, -2)
                loss = self.loss(logits, y)
            
            accuracy = accuracy_fn(y, logits)
            return {'loss': float(loss), 'accuracy': float(accuracy)}
    
    def log_scalars(self, log: dict, prefix):
        prefix_copy = {f'{prefix}/{k}': v for k, v in log.items()}
        wandb.log(prefix_copy, step=self.steps)

    def train(self, train_data, eval_data=None):
        """train one epoch"""
        if eval_data is not None:
            eval_data = cycle(eval_data)
            eval_losses: dict = None  # placeholder

        train_losses: dict = None  # placeholder

        metric_accum = 0
        for y in tqdm(train_data, unit='step'):
            metric_accum += 1

            y = y.cuda()
            new_train_log = self.train_step(y)
            if train_losses is None:
                train_losses = new_train_log
            else:
                # average logging across self.metric_step(s)
                self.sum_scalars(train_losses, new_train_log)
            
            if eval_data is not None:
                y = next(eval_data)
                y = y.cuda()
                new_eval_log  = self.eval_step(y)
                if eval_losses is None:
                    eval_losses = new_eval_log
                else:
                    self.sum_scalars(eval_losses, new_eval_log)
            
            # log scalar metrics
            if (self.steps + 1) % self.metric_step == 0:
                self.div_scalars(train_losses, metric_accum)
                self.log_scalars(train_losses, 'train')
                self.log_scalars(self.compute_sparsity(y), 'train')
                self.log_scalars({'a': self.init_a, 'b': self.init_b}, 'train')
                self.zero_scalars(train_losses)
                if eval_data is not None:
                    self.div_scalars(eval_losses, metric_accum)
                    self.log_scalars(eval_losses, 'eval')
                    self.zero_scalars(eval_losses)
                metric_accum = 0
            
            if (self.steps + 1) % self.checkpoint_step == 0:
                self.regulate_checkpoints()
        self.regulate_checkpoints()  # save final checkpoint
    
    def train_epochs(self, epochs, train_data, eval_data=None):
        run = wandb.init(project='MusicGeneration', entity='allanl', dir=self.dir, id=self.id, resume='allow', config=self.config)
        wandb.watch(self.model)
        with run:
            for i in range(epochs):
                self.train(iter(train_data), eval_data)
    
    def log_final_performance(self, data, title):
        run = wandb.init(project='MusicGeneration', entity='allanl', dir=self.dir, id=self.id, resume='allow', config=self.config)
        with run:
            table = wandb.Table(data=data, columns=["Sequence_length", "Perplexity"])
            wandb.log({"model_sequence_extrapolation": wandb.plot.line(table, "Sequence_length", "Perplexity", 
                      title=title)})


def accuracy_fn(real, pred):
    with torch.no_grad():
        accuracies = real.eq(pred.argmax(dim=1)) # pred = [batch, logits, seq_len]
        mask = real.eq(0).logical_not()
        accuracies = accuracies.logical_and(mask)
        return accuracies.float().sum() / mask.float().sum()


@torch.no_grad()
def windowed_sample_pure(model: torch.nn.Module, sequence: torch.Tensor, sample_len: int, window_size=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Samples a new sequence purely from its own predictions
    sequence.shape = [batch, seq_len]
    """
    logits: List[torch.Tensor] = []
    for i in tqdm(range(sample_len)):
        new_logits = model(sequence)
        logits.append(new_logits[:, -1:, :])
        sequence = torch.cat([sequence, logits[-1].argmax(-1)], dim=-1)
        if window_size is not None and window_size > sequence.shape[-1]:
            sequence = sequence[:, 1:]
    return torch.cat(logits, dim=1)


@torch.no_grad()
def calculate_perplexities_pure(model: torch.nn.Module, sequence: torch.Tensor, sample_sizes: List[int], primer_size: int, window=None):
    """
    Uses ground truth of length model_max_sequence_len to predict a portion of length size,
    not teacher forced
    length of sequence must be greater or equal to max(sample_sizes) + model_max_sequence_len
    """
    perplexities = []
    cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=0)
    for size in sample_sizes:
        logits = windowed_sample_pure(model, sequence[:, :primer_size], size, window)
        logits = logits.transpose(-1, -2)
        #loss = cross_entropy(logits, sequence[:, model_max_sequence_len:model_max_sequence_len+size])
        loss = accuracy_fn(sequence[:, primer_size:primer_size+size], logits)
        perplexities.append(float(loss))
    return [[x, y] for (x, y) in zip(sample_sizes, perplexities)]


@torch.no_grad()
def windowed_sample_teacher(model: torch.nn.Module, sequence: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Samples new tokens from previous ground truth tokens of size window_size
    """
    logits: List[torch.Tensor] = [model(sequence[:, :window_size])]
    for i in tqdm(range(1, sequence.shape[-1] - window_size + 1)):
        logits.append(model(sequence[:, i:window_size+i])[:, -1:, :])
    
    return torch.cat(logits, dim=1)


def calculate_perplexities_teacher(model: torch.nn.Module, sequence: torch.Tensor, sample_sizes: List[int], window=None):
    """
    For each size in sample size, calculates a chunk of sequence of length size,
    where each new token is predicted from previous ground truth tokens.
    Optionally, model_max_sequence_len can be a list of integers,
    for models that support sequence lengths that are greater than its training
    sequence length; in this case, sample_sizes would have to equal model_max_sequence_len.
    """
    perplexities = []
    cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=0)
    for size in sample_sizes:
        if window is None:
            logits = windowed_sample_teacher(model, sequence[:, :size], size)
        else:
            logits = windowed_sample_teacher(model, sequence[:, :size], window)
        logits = logits.transpose(-1, -2)
        loss = accuracy_fn(sequence[:, 1:size+1], logits)
        perplexities.append(float(loss))
    return [[x, y] for (x, y) in zip(sample_sizes, perplexities)]
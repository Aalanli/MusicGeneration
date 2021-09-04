import os
import pathlib
import dill
from itertools import cycle
from tqdm import tqdm

import torch
import wandb


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
        criterion: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        directory: str = None,
        metric_step: int = None,
        checkpoint_step: int = None,
        log_results_step: int = None,
        mixed_precision: bool = None,
        lr_scheduler: bool = None,
        max_checkpoints: int = None,
        config=None) -> None:

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.mixed_precison = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
        self.dir = directory
        self.metric_step = metric_step
        self.checkpoint_step = checkpoint_step
        self.log_results_step = log_results_step
        self.config = config

        self.max_checkpoints_saved = max_checkpoints
        self.steps = 0

        self.checkpointable = ['model', 'criterion', 'optimizer', 'scaler', 'steps']
        
        self.id = self.config.name
        self.batch_splits = self.config.batch_splits

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
    
    def accuracy(self, real, pred):
        with torch.no_grad():
            accuracies = real.eq(pred.argmax(dim=1)) # pred = [batch, logits, seq_len]
            mask = real.eq(0).logical_not()
            accuracies = accuracies.logical_and(mask)
            return accuracies.float().sum() / mask.float().sum()

    def train_step(self, sample):
        y = sample[:, 1:]
        x = sample[:, :-1]
        self.model.train()
        for x1, y1 in zip(x.chunk(self.batch_splits), y.chunk(self.batch_splits)):
            with torch.cuda.amp.autocast(enabled=self.mixed_precison):
                #mask = generate_square_subsequent_mask(x1.size()[1])
                logits = self.model(x1)  # logits = [batch, seq_len, classes]
                logits = logits.transpose(-1, -2)  # loss expects logit classes in dim 1
                loss = self.loss(logits, y1)
                loss = loss.div(self.batch_splits)
            
            self.scaler.scale(loss).backward()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(self.steps)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.steps += 1

        accuracy = self.accuracy(y1, logits)
        return {'loss': float(loss), 'accuracy': float(accuracy)}
    
    def eval_step(self, sample):
        with torch.cuda.amp.autocast(enabled=self.mixed_precison):
            with torch.no_grad():
                y = sample[:, 1:]
                x = sample[:, :-1]
                self.model.eval()
                #mask = generate_square_subsequent_mask(x.size()[-1])

                logits = self.model(x)
                logits = logits.transpose(-1, -2)
                loss = self.loss(logits, y)
            
            accuracy = self.accuracy(y, logits)
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
        for y in tqdm(train_data, desc='epoch', unit='step'):
            metric_accum += 1

            y = y.cuda()
            new_train_log, model_out = self.train_step(y)
            if train_losses is None:
                train_losses = new_train_log
            else:
                # average logging across self.metric_step(s)
                self.sum_scalars(train_losses, new_train_log)
            
            if eval_data is not None:
                y = next(eval_data)
                y = y.cuda()
                new_eval_log, _ = self.eval_step(y)
                if eval_losses is None:
                    eval_losses = new_eval_log
                else:
                    self.sum_scalars(eval_losses, new_eval_log)
            
            # log scalar metrics
            if (self.steps + 1) % self.metric_step == 0:
                self.div_scalars(train_losses, metric_accum)
                self.log_scalars(train_losses, 'train')
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
        run = wandb.init(project='ARTR', entity='allanl', dir=self.dir, id=self.id, resume='allow', config=self.config)
        wandb.watch(self.model)
        with run:
            for i in range(epochs):
                self.train(iter(train_data), eval_data)
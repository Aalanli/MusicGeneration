import torch
from torch import nn
from torch.utils import checkpoint


def checkpoint_wrapper(module, *args, apply=False, over_ride=None):
    def inner(*x):
        return module(*x)
            
    if over_ride is True:
        return checkpoint.checkpoint(inner, *args)
    if over_ride is False:
        return inner(*args)
    if checkpoint is True:
        return checkpoint.checkpoint(inner, *args)
    return inner(*args)


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class Norm(nn.Module):
    def __init__(self, n_state, axis=-1, epsilon=1e-5):
        super().__init__()

        self.n_state = n_state
        self.g = nn.Parameter(torch.ones([self.n_state]))
        self.b = nn.Parameter(torch.zeros([self.n_state]))

        self.axis = axis
        self.epsilon = epsilon
    
    def forward(self, x):
        u = torch.mean(x, dim=self.axis, keepdim=True)
        s = torch.mean(torch.square(x - u), dim=self.axis, keepdim=True)
        x = (x - u) * torch.rsqrt(s + self.epsilon)
        x = x * self.g + self.b
        return x


class Conv1d(nn.Module):
    def __init__(self, nf, nx, stdev=0.02):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.stdev = stdev

        self.w = nn.Parameter(torch.normal(size=[1, self.nx, self.nf], mean=0.0, std=self.stdev))
        self.b = nn.Parameter(torch.zeros([self.nf]))
    
    def forward(self, x: torch.Tensor):
        shape = x.size()
        start, nx = shape[:-1], shape[-1]
        return torch.reshape(torch.matmul(torch.reshape(x, [-1, nx]), torch.reshape(self.w, [-1, self.nf])) + self.b, start + (self.nf,))


class Attention(nn.Module):
    def __init__(self, heads, n_state):
        super().__init__()
        assert n_state % heads == 0
        
        self.heads = heads
        self.n_state = n_state
        self.depth = self.n_state // self.heads

    def split_heads(self, x: torch.Tensor, batch: int, seq_len: int):
        # size = [batch, sequence, features]
        # split features into heads; size = [batch, heads, sequence, depth]

        x = x.reshape((batch, seq_len, self.heads, self.depth))
        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x: torch.Tensor, batch: int, seq_len: int):
        # inverse operation of split heads
        x = x.permute(0, 2, 1, 3)
        return x.reshape((batch, seq_len, self.n_state))


class Mlp(nn.Module):
    def __init__(self, input_dim, proj_dim):
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        
        self.conv_fc = Conv1d(self.proj_dim, self.input_dim)
        self.conv_proj = Conv1d(self.input_dim, self.proj_dim)
    
    def forward(self, x):
        h = nn.functional.gelu(self.conv_fc(x))
        return self.conv_proj(h)


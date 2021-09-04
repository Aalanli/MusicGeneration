import math
import torch
import torch.nn as nn
from base_layers import *


class SimpleAttention(Attention):
    def __init__(self, heads, n_state, max_sequence):
        super().__init__(heads, n_state, max_sequence)
    
    def multihead_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        """
        Most naive implementation
        mask.shape = [bs, k.shape[-2]]; k.shape[-2] = k_seq_len
        """
        depth = q.shape[-1]
        w = q @ k.transpose(-1, -2)
        w = w / math.sqrt(depth)

        if mask is not None:
            w = w.masked_fill(mask, float('-inf'))
        
        a = w.softmax(-1)
        out = a @ v
        return out, a
    


class FastDecoderBlock(nn.Module):
    def __init__(self, heads, max_sequence, n_state, checkpoint, projection_features=None):
        super().__init__()
        self.heads = heads
        self.max_sequence = max_sequence
        self.n_state = n_state
        self.checkpoint = checkpoint

        self.attn = SimpleAttention(heads, n_state, max_sequence)
        self.norm1 = Norm(self.n_state)
        self.norm2 = Norm(self.n_state)
        self.mlp = Mlp(self.n_state, self.n_state * 4)
    
    def forward(self, x, pos_emb):
        a = checkpoint_wrapper(self.norm1, x, apply=self.checkpoint, over_ride=None)
        a = checkpoint_wrapper(self.attn, a, pos_emb, apply=self.checkpoint, over_ride=None)
        x = x + a
        m = checkpoint_wrapper(self.norm2, x, apply=self.checkpoint, over_ride=None)
        m = checkpoint_wrapper(self.mlp, m, apply=self.checkpoint, over_ride=None)
        x = x + m
        return x

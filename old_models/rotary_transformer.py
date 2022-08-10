# %%
import math
import torch
import torch.nn as nn
from base_layers import *
from einops import rearrange, repeat


def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(q, k, sinu_pos):
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j = 2)
    sin, cos = sinu_pos.unbind(dim = -2)
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j = 2), (sin, cos))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.dim = dim
        self.inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('emb', emb)
    
    def calculate_new_embed(self, x):
        position = torch.arange(0, x.shape[1], dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, self.inv_freq)
        return torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1).unsqueeze(0).to(x)
    
    def forward(self, x):
        return self.emb[None, :x.shape[1], :].to(x)


class SimpleAttention(Attention):
    def __init__(self, heads, n_state):
        super().__init__(heads, n_state)
        self.c_attn = Conv1d(self.n_state * 3, self.n_state)
        self.c_proj = Conv1d(self.n_state, self.n_state)
    
    def multihead_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        """
        Most naive implementation
        mask.shape = [bs, k.shape[-2]]; k.shape[-2] = k_seq_len
        """
        depth = q.shape[-1]
        w = q @ k.transpose(-1, -2)
        w = w / math.sqrt(depth)

        if mask is not None:
            w = w + mask
        
        a = w.softmax(-1)
        out = a @ v
        return out
    
    def forward(self, x: torch.Tensor, mask, pos):
        batch, seq_len, _ = x.size()
        c = self.c_attn(x)
        q, k, v = torch.split(c, self.n_state, dim=2)
        q = self.split_heads(q, batch, seq_len)
        k = self.split_heads(k, batch, seq_len)
        v = self.split_heads(v, batch, seq_len)

        q, k = apply_rotary_pos_emb(q, k, pos)

        a = self.multihead_attn(q, k, v, mask)
        a = self.combine_heads(a, batch, seq_len)
        a = self.c_proj(a)
        return a


class DecoderBlock(nn.Module):
    def __init__(self, heads, n_state, proj_forward, activation, dropout):
        super().__init__()
        self.heads = heads
        self.n_state = n_state

        self.attn = SimpleAttention(heads, n_state)

        self.linear1 = nn.Linear(n_state, proj_forward)
        self.linear2 = nn.Linear(proj_forward, n_state)
        self.norm1 = nn.LayerNorm(n_state)
        self.norm2 = nn.LayerNorm(n_state)
        self.drop = nn.Dropout(dropout)
        self.activation = activation
    
    def forward(self, x, mask, pos):
        x1 = self.attn(x, mask, pos)
        x = self.drop(x1) + x  # save, non-deterministic
        x = self.norm1(x)
        x1 = self.activation(self.linear1(x))
        x1 = self.drop(x1)  # save, non-deterministic
        x1 = self.linear2(x1)
        x = x + x1
        x = self.norm2(x)
        return x


class RotaryTransformer(nn.Module):
    def __init__(self, n_vocab, d_model, n_layers, n_heads, max_sequence, proj_forward, dropout=0.1, activation=torch.nn.functional.relu) -> None:
        super().__init__()

        self.n_vocab = n_vocab
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = n_heads
        self.max_sequence = max_sequence
        
        self.embedding = nn.Parameter(torch.normal(size=(self.n_vocab, self.d_model), mean=0.0, std=0.02)) 
        self.decoder_layers = nn.ModuleList([DecoderBlock(self.heads, d_model, proj_forward, activation, dropout) for _ in range(self.n_layers)])
        self.norm = Norm(self.d_model)
        self.pos_embed = FixedPositionalEmbedding(d_model // n_heads, max_sequence)
        mask = generate_square_subsequent_mask(max_sequence, 'cpu')
        self.register_buffer("mask", mask)
    
    def forward(self, x: torch.Tensor):
        batch, seq_len = x.size()
        if seq_len > self.max_sequence:
            mask = generate_square_subsequent_mask(seq_len, x.device)
            pos = self.pos_embed.calculate_new_embed(x)
        else:
            mask = self.mask[:seq_len, :seq_len]
            pos = self.pos_embed(x)

        h = self.embedding[x]

        for layer in range(self.n_layers):
            h = self.decoder_layers[layer](h, mask, pos)

        h = self.norm(h)
        h_flat = h.reshape(batch * seq_len, self.d_model)
        logits = h_flat @ self.embedding.transpose(-1, -2)
        logits = logits.reshape([batch, seq_len, self.n_vocab])

        return logits

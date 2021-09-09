# %%
import math
import torch
import torch.nn as nn

from base_layers import *


def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
    else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround. 
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]


def construct_alibi(max_seq_len, attn_heads):
    slopes = torch.Tensor(get_slopes(attn_heads))
    alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).expand(attn_heads, -1, -1)
    mask = generate_square_subsequent_mask(max_seq_len, 'cpu')
    mask = mask.unsqueeze(0) + alibi
    return mask


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
    
    def forward(self, x: torch.Tensor, mask):
        batch, seq_len, _ = x.size()
        c = self.c_attn(x)
        q, k, v = torch.split(c, self.n_state, dim=2)
        q = self.split_heads(q, batch, seq_len)
        k = self.split_heads(k, batch, seq_len)
        v = self.split_heads(v, batch, seq_len)

        a = self.multihead_attn(q, k, v, mask)
        a = self.combine_heads(a, batch, seq_len)
        a = self.c_proj(a)
        return a


class DecoderBlock(nn.Module):
    def __init__(self, heads, n_state, proj_forward, activation, dropout):
        super().__init__()
        self.heads = heads
        self.n_state = n_state
        self.checkpoint = checkpoint

        self.attn = SimpleAttention(heads, n_state)

        self.linear1 = nn.Linear(n_state, proj_forward)
        self.linear2 = nn.Linear(proj_forward, n_state)
        self.norm1 = nn.LayerNorm(n_state)
        self.norm2 = nn.LayerNorm(n_state)
        self.drop = nn.Dropout(dropout)
        self.activation = activation
    
    def forward(self, x, mask):
        x1 = self.attn(x, mask)
        x = self.drop(x1) + x  # save, non-deterministic
        x = self.norm1(x)
        x1 = self.activation(self.linear1(x))
        x1 = self.drop(x1)  # save, non-deterministic
        x1 = self.linear2(x1)
        x = x + x1
        x = self.norm2(x)
        return x


class AlibiTransformer(nn.Module):
    def __init__(self, n_vocab, d_model, n_layers, n_heads, max_sequence, proj_forward, dropout=0.1, activation=torch.nn.functional.relu) -> None:
        super().__init__()

        self.n_vocab = n_vocab
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = n_heads
        self.max_sequence = max_sequence
        
        self.embedding = nn.Parameter(torch.normal(size=(self.n_vocab, self.d_model), mean=0.0, std=0.02)) 
        self.decoder_layers = nn.ModuleList([DecoderBlock(n_heads, d_model, proj_forward, activation, dropout) for _ in range(self.n_layers)])
        self.norm = Norm(self.d_model)

        mask = construct_alibi(max_sequence, n_heads)
        self.register_buffer("mask", mask)
    
    def forward(self, x: torch.Tensor):
        batch, seq_len = x.size()
        if seq_len > self.max_sequence:
            mask = construct_alibi(seq_len, self.heads).to(x)
        else:
            mask = self.mask[:, :seq_len, :seq_len]

        h = self.embedding[x]

        for layer in range(self.n_layers):
            h = self.decoder_layers[layer](h, mask)

        h = self.norm(h)
        h_flat = h.reshape(batch * seq_len, self.d_model)
        logits = h_flat @ self.embedding.transpose(-1, -2)
        logits = logits.reshape([batch, seq_len, self.n_vocab])

        return logits

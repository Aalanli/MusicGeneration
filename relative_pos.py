# %%
import torch
import torch.nn as nn

from base_layers import *


class RelativeAttention(Attention):
    def __init__(self, heads, n_state, max_sequence):
        super().__init__(heads, n_state)
        self.max_sequence = max_sequence

        self.c_attn = Conv1d(self.n_state * 3, self.n_state)
        self.c_proj = Conv1d(self.n_state, self.n_state)
        self.E = nn.Parameter(torch.Tensor(self.heads, self.max_sequence, n_state // heads))
        nn.init.xavier_normal_(self.E)

    def relative_attn(self, q: torch.Tensor, E: torch.Tensor, batch: int, seq_len: int):
        # q.size() = [batch, heads, sequence, features]
        q_ = q.permute(1, 0, 2, 3)
        q_ = q_.reshape(self.heads, batch * seq_len, self.depth)

        E = E[:, self.max_sequence - seq_len:]
        rel = q_ @ E.transpose(-1, -2)
        rel = rel.reshape(self.heads, batch, seq_len, seq_len)
        rel = torch.nn.functional.pad(rel, (1, 0), "constant", 0)

        rel = rel.reshape(self.heads, batch, seq_len + 1, seq_len)
        rel = rel[:, :, 1:]

        rel = rel.permute(1, 0, 2, 3)
        return rel
    
    def multihead_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, batch, seq_len, mask=None):
        w = q @ k.transpose(-1, -2)
        w = w + self.relative_attn(q, self.E, batch, seq_len)
        w = w * (1 / (self.depth) ** (1/2))

        if mask is not None:
            w += mask
        
        w = w.softmax(-1)
        a = w @ v
        return a
    
    def forward(self, x: torch.Tensor, mask=None):
        batch, seq_len, _ = x.size()

        c = self.c_attn(x)
        q, k, v = torch.split(c, self.n_state, dim=2)
        q = self.split_heads(q, batch, seq_len)
        k = self.split_heads(k, batch, seq_len)
        v = self.split_heads(v, batch, seq_len)

        a = self.multihead_attn(q, k, v, batch, seq_len, mask)
        a = self.combine_heads(a, batch, seq_len)
        a = self.c_proj(a)
        return a


class DecoderBlock(nn.Module):
    def __init__(self, heads, max_sequence, n_state, proj_forward, activation, dropout):
        super().__init__()
        
        self.heads = heads
        self.max_sequence = max_sequence
        self.n_state = n_state

        self.attn = RelativeAttention(self.heads, self.n_state, self.max_sequence)
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


class RelativeTransformer(nn.Module):
    def __init__(self, n_vocab, d_model, n_layers, n_heads, max_sequence, proj_forward, dropout=0.1, activation=torch.nn.functional.relu) -> None:
        super().__init__()

        self.n_vocab = n_vocab
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = n_heads
        self.max_sequence = max_sequence

        self.embedding = nn.Parameter(torch.normal(size=(self.n_vocab, self.d_model), mean=0.0, std=0.02))
        
        self.decoder_layers = nn.ModuleList([DecoderBlock(n_heads, max_sequence, d_model, proj_forward, activation, dropout) for _ in range(self.n_layers)])
        self.norm = Norm(self.d_model)

        mask = generate_square_subsequent_mask(max_sequence, 'cpu')
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor):
        batch, seq_len = x.size()
        mask = self.mask[:seq_len, :seq_len]

        h = self.embedding[x]

        for layer in range(self.n_layers):
            h = self.decoder_layers[layer](h, mask)

        h = self.norm(h)
        h_flat = h.reshape(batch * seq_len, self.d_model)
        logits = h_flat @ self.embedding.transpose(-1, -2)
        logits = logits.reshape([batch, seq_len, self.n_vocab])

        return logits


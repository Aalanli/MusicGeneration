import torch
import torch.nn as nn

from base_layers import *

class RelativeAttention(Attention):
    def __init__(self, heads, n_state, max_sequence, ):
        super().__init__(heads, n_state, max_sequence)

        self.c_attn = Conv1d(self.n_state * 3, self.n_state)
        self.c_proj = Conv1d(self.n_state, self.n_state)
        self.E = nn.Parameter(torch.Tensor(self.heads, self.max_sequence, n_state))
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
        
        w = torch.nn.functional.softmax(w, dim=-1)
        a = w @ v
        return a
    
    def forward(self, x: torch.Tensor, mask=None, past=None):
        batch, seq_len, _ = x.size()

        c = self.c_attn(x)
        q, k, v = torch.split(c, self.n_state, dim=2)
        q = self.split_heads(q, batch, seq_len)
        k = self.split_heads(k, batch, seq_len)
        v = self.split_heads(v, batch, seq_len)

        present = torch.stack([k, v], dim=1)
        if past is not None:
            pk, pv = torch.unbind(past, dim=1)
            k = torch.cat([pk, k], dim=-2)
            v = torch.cat([pv, v], dim=-2)

        a = self.multihead_attn(q, k, v, batch, seq_len, mask)
        a = self.combine_heads(a, batch, seq_len)
        a = self.c_proj(a)
        return a, present


class DecoderBlock(nn.Module):
    def __init__(self, heads, max_sequence, n_state, checkpoint=True):
        super().__init__()
        
        self.heads = heads
        self.max_sequence = max_sequence
        self.n_state = n_state
        self.checkpoint = checkpoint

        self.attn = RelativeAttention(self.heads, self.n_state, self.max_sequence)
        self.norm1 = Norm(self.n_state)
        self.norm2 = Norm(self.n_state)
        self.mlp = Mlp(self.n_state, self.n_state * 4)

    def forward(self, x, mask=None, past=None):
        a = checkpoint_wrapper(self.norm1, x, apply=self.checkpoint, over_ride=None)
        a, present = checkpoint_wrapper(self.attn, a, mask, past, apply=self.checkpoint, over_ride=None)
        x = x + a
        m = checkpoint_wrapper(self.norm2, x, apply=self.checkpoint, over_ride=None)
        m = checkpoint_wrapper(self.mlp, m, apply=self.checkpoint, over_ride=None)
        x = x + m
        return x, present


class RelativeTransformer(nn.Module):
    def __init__(self, n_vocab, d_model, n_layers, n_heads, max_sequence, checkpoint=False):
        super().__init__()

        self.n_vocab = n_vocab
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = n_heads
        self.max_sequence = max_sequence

        self.embedding = nn.Parameter(torch.normal(size=(self.n_vocab, self.d_model), mean=0.0, std=0.02))
        
        self.decoder_layers = nn.ModuleList([DecoderBlock(self.heads, self.max_sequence, self.d_model, checkpoint=checkpoint) for _ in range(self.n_layers)])
        self.norm = Norm(self.d_model)

    def forward(self, x: torch.Tensor, mask=None, past=None):
        batch, seq_len = x.size()

        h = self.embedding[x]

        for layer in range(self.n_layers):
            h, present = self.decoder_layers[layer](h, mask, past)
            del present

        h = self.norm(h)
        h_flat = h.reshape(batch * seq_len, self.d_model)
        logits = h_flat @ self.embedding.transpose(-1, -2)
        logits = logits.reshape([batch, seq_len, self.n_vocab])

        return logits
    
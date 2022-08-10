# %%
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from base_layers import *

from tqdm import tqdm
from torchmetrics import Metric, Accuracy


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
    def __init__(self, 
        vocab_note, vocab_velocity, vocab_duration, 
        embed_note, embed_velocity, embed_duration, 
        n_layers, n_heads, max_sequence, proj_forward, dropout=0.1, 
        activation='gelu',
        **kwargs) -> None:
        super().__init__()

        self.vocab_note = vocab_note
        self.vocab_velocity = vocab_velocity
        self.vocab_duration = vocab_duration
        self.dim_note = embed_note
        self.dim_velocity = embed_velocity
        self.dim_duration = embed_duration
        self.d_model = embed_note + embed_velocity + embed_duration
        self.n_layers = n_layers
        self.heads = n_heads
        self.max_sequence = max_sequence
        
        self.embed_note = nn.Embedding(vocab_note, embed_note)
        self.embed_velocity = nn.Embedding(vocab_velocity, embed_velocity)
        self.embed_duration = nn.Embedding(vocab_duration, embed_duration)

        activation = getattr(torch.nn.functional, activation)
        self.decoder_layers = nn.ModuleList([DecoderBlock(n_heads, self.d_model, proj_forward, activation, dropout) for _ in range(self.n_layers)])
        self.norm = Norm(self.d_model)

        self.proj_note     = nn.Linear(self.d_model, vocab_note)
        self.proj_velocity = nn.Linear(self.d_model, vocab_velocity)
        self.proj_duration = nn.Linear(self.d_model, vocab_duration)

        mask = construct_alibi(max_sequence, n_heads)
        self.register_buffer("mask", mask)
    
    def forward(self, x: torch.Tensor):
        batch, _, seq_len = x.size()
        note_embed = self.embed_note(x[:, 0])
        velocity_embed = self.embed_velocity(x[:, 1])
        duration_embed = self.embed_duration(x[:, 2])

        h = torch.concat([note_embed, velocity_embed, duration_embed], dim=-1)

        if seq_len > self.max_sequence:
            mask = construct_alibi(seq_len, self.heads).to(x)
        else:
            mask = self.mask[:, :seq_len, :seq_len]

        for layer in range(self.n_layers):
            h = self.decoder_layers[layer](h, mask)

        h = self.norm(h)
        
        logit_n = self.proj_note(h)
        logit_v = self.proj_velocity(h)
        logit_d = self.proj_duration(h)
        return logit_n, logit_v, logit_d
    
    def inference(self, seq: torch.Tensor, idx=0, deterministic=False):
        with torch.no_grad():            
            for i in tqdm(range(idx, seq.shape[-1])):
                n, v, d = self(seq)
                if deterministic:
                    r = map(lambda x: torch.multinomial(x[:, -1].softmax(-1), num_samples=1), [n, v, d])
                else:
                    r = map(lambda x: torch.argmax(x[:, -1], -1), [n, v, d])
                r = torch.stack(list(r), -1)
                seq[:, :, i] = r
        return seq


class Criterion(nn.Module):
    def __init__(self,
        eos_coef=0.1,
        vocab=[],
        eos_tokens=[0, 1],
        weights={},
        **kwargs) -> None:

        super().__init__()
        self.loss_weights = weights
        
        cls_weights = [torch.ones(n) for n in vocab]
        for i in range(len(vocab)):
            for j in eos_tokens:
                cls_weights[i][j] = eos_coef
        self.register_buffer('weight_n', cls_weights[0])
        self.register_buffer('weight_v', cls_weights[1])
        self.register_buffer('weight_d', cls_weights[2])

    def forward(self, xs, y):
        n, v, d = xs
        losses = {}
        losses['loss_notes']    = F.cross_entropy(n.transpose(-1, -2), y[:, 0], self.weight_n)
        losses['loss_velocity'] = F.cross_entropy(v.transpose(-1, -2), y[:, 1], self.weight_v)
        losses['loss_duration'] = F.cross_entropy(d.transpose(-1, -2), y[:, 2], self.weight_d)
        loss = sum([self.loss_weights[k] * losses[k] for k in losses])
        losses['loss'] = loss
        return loss, losses

class Metrics(Metric):
    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.acc_n = Accuracy(mdmc_reduce='samplewise')
        self.acc_v = Accuracy(mdmc_reduce='samplewise')
        self.acc_d = Accuracy(mdmc_reduce='samplewise')
        self.total_acc = Accuracy(mdmc_reduce='samplewise')
    
    def update(self, xs, y):
        n, v, d = xs
        self.acc_n.update(n, y[:, 0])
        self.acc_v.update(v, y[:, 1])
        self.acc_d.update(d, y[:, 2])
        self.total_acc.update(torch.stack([n, v, d], 1), y)
    
    def compute(self):
        acc = {}
        acc['acc_notes'] = self.acc_n.compute()
        acc['acc_velocity'] = self.acc_v.compute()
        acc['acc_duration'] = self.acc_d.compute()
        acc['accuracy'] = self.total_acc.compute()
        return acc
    
    def reset(self):
        self.acc_n.reset()
        self.acc_v.reset()
        self.acc_d.reset()
        self.total_acc.reset()

        

def build_model_and_criterion(args):
    c = Criterion(vocab=[args.vocab_note, args.vocab_velocity, args.vocab_duration], **args)
    return AlibiTransformer(**args), c


# %%
if __name__ == '__main__':
    import torch
    from torchmetrics import Accuracy
    acc_d = Accuracy(mdmc_reduce='samplewise')
    a = torch.randint(0, 19, [4, 512])
    b = torch.randint(0, 19, [4, 512])
    print(torch.stack([a, b], 1).shape)
    acc_d.update(a, b)

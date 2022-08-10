# %%
import functools

from typing import Callable
import math

import jax
from jax import numpy as jnp, lax, random
import flax
from flax import linen as nn

def masked_fill(mask, a, fill):
    return jax.lax.select(mask, a, jax.lax.broadcast(fill, a.shape))

def generate_square_subsequent_mask(sz):
    mask = ~(jnp.triu(jnp.ones((sz, sz), jnp.int32)) == 1)
    mask = mask.astype(jnp.float32).T
    mask = masked_fill(mask == 0, mask, float('-inf'))
    return mask

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

@functools.partial(jax.jit, static_argnums=[0, 1])
def construct_alibi(max_seq_len, attn_heads):
    slopes = jnp.array(get_slopes(attn_heads))
    slopes = jnp.expand_dims(slopes, (1, 2))
    alibi = slopes * jnp.repeat(jnp.expand_dims(jnp.arange(0, max_seq_len), (0, 1)), attn_heads, 0)
    
    mask = generate_square_subsequent_mask(max_seq_len)
    mask = jnp.expand_dims(mask, (0,)) + alibi
    return mask

@jax.jit
def attn(q, k, v, mask=None):
    depth = q.shape[-1]
    w = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2)))
    w = w / jnp.sqrt(depth)
    if mask is not None:
        w = w + mask[None]
    a = jax.nn.softmax(w, -1)
    out = jnp.matmul(a, v)
    return out

@functools.partial(jax.jit, static_argnums=3)
def chunked_attn(q, k, v, block_n, mask=None):
    b, h, l, d = q.shape
    k = jnp.transpose(k, (0, 1, 3, 2)) # b, h, d, l
    y = jnp.zeros_like(v)

    for i in range(0, l-block_n, block_n):
        a = jnp.matmul(q[:, :, i:i+block_n, :], k) / jnp.sqrt(d)
        a += mask[i:i+block_n]
        a = jax.nn.softmax(a)
        h = jnp.matmul(a, v)
        y = jax.lax.dynamic_update_slice(y, h, (0, 0, i, 0))
    return y


def casual_conv(features, kernel_size):
    return nn.linear.Conv(features, kernel_size, padding=(kernel_size-1, 0))


class MultiHeadAttn(nn.Module):
    heads: int
    @nn.compact
    def __call__(self, x, mask=None):
        b, l, d = x.shape
        x = nn.Dense(d * 3)(x)
        x = jnp.reshape(x, (b, l, 3, self.heads, -1))
        x = jnp.transpose(x, (0, 2, 3, 1, 4))
        q, k, v = x[:, 0], x[:, 1], x[:, 2]
        y = attn(q, k, v, mask)
        y = jnp.reshape(jnp.transpose(y, (0, 2, 1, 3)), (b, l, d))
        return y

class AttnProj(nn.Module):
    heads: int
    expansion: int = 4
    dropout: int = 0.1
    act: Callable = nn.activation.gelu
    @nn.compact
    def __call__(self, x, mask=None):
        d = x.shape[-1]
        x1 = MultiHeadAttn(self.heads)(x, mask)
        x = nn.Dropout(self.dropout, deterministic=True)(x1) + x
        x = nn.LayerNorm()(x) 
        x1 = self.act(nn.Dense(d * self.expansion)(x))
        x1 = nn.Dropout(self.dropout, deterministic=True)(x1)
        x1 = nn.Dense(d)(x)
        x = x + x1
        x = nn.LayerNorm()(x)
        return x

class AlibiTransformer(nn.Module):
    heads: int
    vocab: int
    d_model: int
    n_layers: int
    def setup(self):
        self.layers = [AttnProj(self.heads) for i in range(self.n_layers)]
        self.embed = nn.Embed(self.vocab, self.d_model)
        self.out_proj = nn.Dense(self.vocab)

    def __call__(self, x, mask=None):
        l = x.shape[-1]
        if mask is None:
            mask = construct_alibi(l, self.heads)
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h, mask)
        logits = self.out_proj(h)
        return logits


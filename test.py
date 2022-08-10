# %%
import jax
from jax import numpy as jnp

a = jnp.arange(0, 50)
a = jnp.reshape(a, (2, 5, 5))
ind = jnp.array([[0, 0, 1, 2, 1], [0, 0, 0, 0, 0]])
index_row = jax.vmap(lambda a, b: a[b], -1)
index_batch = jax.vmap(index_row, 0)

h = index_batch(a, ind)
print(a)
print(h)

# %%
import torch

a = torch.rand(4, 127, 128)
f = torch.nn.Conv1d(128, 32, 1)
print(f(a).shape)

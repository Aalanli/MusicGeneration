# %%
import jax
import jax.numpy as jnp
import optax

@jax.jit
def loss_fn(logits, truth):
    one_hot = jax.nn.one_hot(truth, logits.shape[-1])
    print(one_hot.shape)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss

@jax.jit
def loss_fn(logits, truth):
    logits = jax.nn.log_softmax(logits, -1)
    index_row = jax.vmap(lambda a, b: a[b], 0)
    index_batch = jax.vmap(index_row, 0)
    logits = index_batch(logits, truth)
    return -jnp.mean(logits)

index_row = jax.vmap(lambda a, b: a[b], 0)
index_batch = jax.vmap(index_row, 0)

a = jax.random.normal(jax.random.PRNGKey(1), (4, 512, 512))
b = jax.random.randint(jax.random.PRNGKey(1), [4, 512], 0, 511)

# %timeit loss_fnv2(a, b)
# %timeit loss_fn(a, b)

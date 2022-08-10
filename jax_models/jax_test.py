# %%
from typing import Callable

import jax
from jax import numpy as jnp, lax, random
import flax
from flax import linen as nn

class SimpleDense(nn.Module):
    features: int
    in_features: int

    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros
        
    def setup(self) -> None:
        self.kernel = self.param('kernel',
                            self.kernel_init, # Initialization function
                            (self.in_features, self.features))  # shape info.
        self.bias = self.param('bias', self.bias_init, (self.features,))

    def __call__(self, inputs):
        kernel = self.kernel

        y = lax.dot_general(inputs, kernel,
                            (((inputs.ndim - 1,), (0,)), ((), ())),) # TODO Why not jnp.dot?
        
        y = y + self.bias
        return y

key1, key2 = random.split(random.PRNGKey(0), 2)
x = random.uniform(key1, (4,4))

model = SimpleDense(features=3, in_features=4)
params = model.init(key2, x)
y = model.apply(params, x)

print('initialized parameters:\n', params)
print('output:\n', y)

# %%
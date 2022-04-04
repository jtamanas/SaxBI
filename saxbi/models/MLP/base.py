import jax
import jax.numpy as np

import flax.linen as nn


class ResidualBlock(nn.Module):
    hidden_dim: int
    act: str = "celu"

    @nn.compact
    def __call__(self, x):
        y = nn.Dense(self.hidden_dim)(x)
        y = getattr(nn, self.act)(y)
        y = nn.Dense(x.shape[-1])(y)
        return x + y


class MLP(nn.Module):
    # TODO: replace with Sequential
    output_dim: int
    num_layers: 5
    hidden_dim: 128
    use_residual: bool = False
    act: str = "celu"

    @nn.compact
    def __call__(self, *args):
        x = np.hstack(args)
        for layer in range(self.num_layers):
            if self.use_residual:
                x = ResidualBlock(self.hidden_dim, self.act)(x)
            else:
                x = nn.Dense(self.hidden_dim)(x)
                x = getattr(nn, self.act)(x)
        return nn.Dense(self.output_dim)(x)
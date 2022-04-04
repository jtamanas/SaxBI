import operator
from flax.linen.module import compact
import jax
import jax.numpy as np
import flax.linen as nn
from typing import Any


class ActNorm(nn.Module):
    """
    Mostly copied from
    https://www.kaggle.com/ameroyer/introduction-to-glow-generative-model-in-jax
    """
    scale: float = 1.0
    eps: float = 1e-8

    @compact
    def __call__(self, inputs, context=None, reverse=False):
        axes = tuple(i for i in range(len(inputs.shape) - 1))

        def dd_mean_initializer(key, shape):
            """Data-dependent init for mu"""
            nonlocal inputs
            x_mean = np.mean(inputs, axis=axes, keepdims=True)
            return -x_mean

        def dd_stddev_initializer(key, shape):
            """Data-dependent init for sigma"""
            nonlocal inputs
            x_var = np.mean(inputs ** 2, axis=axes, keepdims=True)
            var = self.scale / (np.sqrt(x_var) + self.eps)
            return var

        shape = (1,) * len(axes) + (inputs.shape[-1],)
        mu = self.param("actnorm_mean", dd_mean_initializer, shape)
        sigma = self.param("actnorm_stddev", dd_stddev_initializer, shape)

        logsigma = np.log(np.abs(sigma))
        log_det_jacobian = np.prod(np.array(inputs.shape[1:-1])) * np.sum(logsigma)

        if reverse:
            outputs = inputs / (sigma + self.eps) - mu
            log_det_jacobian = -log_det_jacobian
        else:
            outputs = sigma * (inputs + mu)
            log_det_jacobian = log_det_jacobian

        return outputs, log_det_jacobian

    def forward(self, inputs, context=None):
        return self(inputs, context=context, reverse=False)

    def inverse(self, inputs, context=None):
        return self(inputs, context=context, reverse=True)

import jax
import jax.numpy as np
import flax.linen as nn
from typing import Any

Array = Any


class Reverse(nn.Module):
    rng: jax.random.PRNGKey
    input_dim: int

    def __call__(self, inputs, *args, **kwargs):
        perm = np.arange(inputs.shape[-1])[::-1]
        return inputs[:, perm], np.zeros(inputs.shape[:1])

    def forward(self, inputs, *args, **kwargs):
        return self(inputs, *args, **kwargs)

    def inverse(self, inputs, *args, **kwargs):
        perm = np.arange(inputs.shape[-1])[::-1]
        return inputs[:, perm], np.zeros(inputs.shape[:1])


class Random(nn.Module):
    """
    Probably best to use different rng's for each permutation
    """
    rng: jax.random.PRNGKey
    input_dim: int

    def setup(self):
        self.perm = jax.random.permutation(self.rng, np.arange(self.input_dim))

    def __call__(self, inputs, context=None):
        return inputs[:, self.perm], np.zeros(inputs.shape[:1])

    def forward(self, inputs, context=None):
        self(inputs, context=None)

    def inverse(self, inputs, context=None):
        inverse_perm = np.argsort(self.perm)
        return inputs[:, inverse_perm], np.zeros(inputs.shape[:1])


class Conv1x1(nn.Module):
    """
    Mostly taken from
    https://www.kaggle.com/ameroyer/introduction-to-glow-generative-model-in-jax
    """

    rng: jax.random.PRNGKey
    input_dim: int

    def setup(self):
        """Initialize P, L, U, s"""
        # W = PL(U + s)
        # Based on https://github.com/openai/glow/blob/master/model.py#L485
        dim = self.input_dim
        # Sample random rotation matrix
        q, _ = np.linalg.qr(jax.random.normal(self.rng, (dim, dim)), mode="complete")
        p, l, u = jax.scipy.linalg.lu(q)
        # Fixed Permutation (non-trainable)
        self.P = p
        self.P_inv = jax.scipy.linalg.inv(p)
        # Init value from LU decomposition
        L_init = l
        U_init = np.triu(u, k=1)
        s = np.diag(u)
        self.sign_s = np.sign(s)
        S_log_init = np.log(np.abs(s))
        self.l_mask = np.tril(np.ones((dim, dim)), k=-1)
        self.u_mask = np.transpose(self.l_mask)
        # Define trainable variables
        self.L = self.param("L", lambda k, sh: L_init, (dim, dim))
        self.U = self.param("U", lambda k, sh: U_init, (dim, dim))
        self.log_s = self.param("log_s", lambda k, sh: S_log_init, (dim,))

    def __call__(self, inputs, context=None, reverse=False):
        dim = self.input_dim
        assert dim == inputs.shape[-1]
        # enforce constraints that L and U are triangular
        # in the LU decomposition
        L = self.L * self.l_mask + np.eye(dim)
        U = self.U * self.u_mask + np.diag(self.sign_s * np.exp(self.log_s))
        log_det_jacobian = np.sum(self.log_s)

        # forward
        if not reverse:
            # lax.conv uses weird ordering: NCHW and OIHW
            W = np.matmul(self.P, np.matmul(L, U))
            outputs = jax.lax.conv(
                inputs[..., None, None], W[..., None, None], (1, 1), "same"
            )
            outputs = outputs.reshape(inputs.shape[0], dim)
            log_det_jacobian = log_det_jacobian
        # inverse
        else:
            W_inv = np.matmul(
                jax.scipy.linalg.inv(U), np.matmul(jax.scipy.linalg.inv(L), self.P_inv)
            )
            outputs = jax.lax.conv(
                inputs[..., None, None],
                W_inv[..., None, None],
                (1, 1),
                "same",
            )
            outputs = outputs.reshape(inputs.shape[0], dim)
            log_det_jacobian = -log_det_jacobian

        return outputs, log_det_jacobian

    def forward(self, inputs, context=None):
        return self(inputs, context=context, reverse=False)

    def inverse(self, inputs, context=None):
        return self(inputs, context=context, reverse=True)


class Conv1x1Image(nn.Module):
    """
    UNTESTED

    Mostly taken from
    https://www.kaggle.com/ameroyer/introduction-to-glow-generative-model-in-jax
    """

    rng: jax.random.PRNGKey
    channels: int

    def setup(self):
        """Initialize P, L, U, s"""
        # W = PL(U + s)
        # Based on https://github.com/openai/glow/blob/master/model.py#L485
        c = self.channels
        # Sample random rotation matrix
        q, _ = np.linalg.qr(jax.random.normal(self.rng, (c, c)), mode="complete")
        p, l, u = jax.scipy.linalg.lu(q)
        # Fixed Permutation (non-trainable)
        self.P = p
        self.P_inv = jax.scipy.linalg.inv(p)
        # Init value from LU decomposition
        L_init = l
        U_init = np.triu(u, k=1)
        s = np.diag(u)
        self.sign_s = np.sign(s)
        S_log_init = np.log(np.abs(s))
        self.l_mask = np.tril(np.ones((c, c)), k=-1)
        self.u_mask = np.transpose(self.l_mask)
        # Define trainable variables
        self.L = self.param("L", lambda k, sh: L_init, (c, c))
        self.U = self.param("U", lambda k, sh: U_init, (c, c))
        self.log_s = self.param("log_s", lambda k, sh: S_log_init, (c,))

    def __call__(self, inputs, logdet=0, reverse=False):
        c = self.channels
        assert c == inputs.shape[-1]
        # enforce constraints that L and U are triangular
        # in the LU decomposition
        L = self.L * self.l_mask + np.eye(c)
        U = self.U * self.u_mask + np.diag(self.sign_s * np.exp(self.log_s))
        log_det_jacobian = np.sum(self.log_s) * inputs.shape[1] * inputs.shape[2]

        # forward
        if not reverse:
            # lax.conv uses weird ordering: NCHW and OIHW
            W = np.matmul(self.P, np.matmul(L, U))
            outputs = jax.lax.conv(
                np.transpose(inputs, (0, 3, 1, 2)), W[..., None, None], (1, 1), "same"
            )
            outputs = np.transpose(outputs, (0, 2, 3, 1))
            log_det_jacobian = log_det_jacobian
        # inverse
        else:
            W_inv = np.matmul(
                jax.scipy.linalg.inv(U), np.matmul(jax.scipy.linalg.inv(L), self.P_inv)
            )
            outputs = jax.lax.conv(
                np.transpose(inputs, (0, 3, 1, 2)),
                W_inv[..., None, None],
                (1, 1),
                "same",
            )
            outputs = np.transpose(outputs, (0, 2, 3, 1))
            log_det_jacobian = -log_det_jacobian

        return outputs, log_det_jacobian

    def forward(self, inputs, context=None):
        return self(inputs, context=context, reverse=False)

    def inverse(self, inputs, context=None):
        return self(inputs, context=context, reverse=True)

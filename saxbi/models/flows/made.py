import jax
import jax.numpy as np
from flax.linen.module import compact
import flax.linen as nn
from saxbi.models.sequential import Sequential

from typing import Any

Array = Any


def get_masks(input_dim, context_dim=0, hidden_dim=64, num_hidden=1):
    masks = []
    degrees = [np.arange(input_dim)]
    input_degrees = np.arange(input_dim)

    for n_h in range(num_hidden + 1):
        degrees += [np.arange(hidden_dim) % (input_dim - 1)]
    degrees += [input_degrees % input_dim - 1]

    for i, (d0, d1) in enumerate(zip(degrees[:-1], degrees[1:])):
        mask = np.transpose(np.expand_dims(d1, -1) >= np.expand_dims(d0, 0)).astype(
            np.float32
        )
        if i == 0:  # pass in context
            # TODO: This still doesn't pass context to the most-masked element.
            #       Need to figure out how to do that effectively.
            mask = np.vstack((mask, np.ones((context_dim, mask.shape[-1]))))
        masks += [mask]

    return masks


class MaskedDense(nn.Dense):
    mask: Array = None
    use_context: bool = False

    @compact
    def __call__(self, inputs: Array, context=None) -> Array:
        """
        Taken from flax.linen.Dense.
        Applies a masked linear transformation to the inputs along the last dimension.

        Args:
        inputs: The nd-array to be transformed.

        Returns:
        The transformed input.
        """
        inputs = np.asarray(inputs, self.dtype)
        if context is not None and self.use_context:
            assert (
                inputs.shape[0] == context.shape[0]
            ), "inputs and context must have the same batch size"
            inputs = np.hstack([inputs, context])

        kernel = self.param(
            "kernel", self.kernel_init, (self.mask.shape[0], self.features)
        )
        kernel = np.asarray(kernel, self.dtype)
        kernel = kernel * self.mask  # major difference from flax.linen.Dense
        y = jax.lax.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,))
            bias = np.asarray(bias, self.dtype)
            y += np.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


class MaskedTransform(nn.Module):
    input_dim: int
    context_dim: int = 0
    hidden_dim: int = 64
    num_hidden: int = 1
    output_dim_multiplier: int = 2
    act: str = "relu"

    def setup(self):
        masks = get_masks(
            self.input_dim,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            num_hidden=self.num_hidden,
        )

        self.layers = [
            MaskedDense(features=masks[0].shape[-1], mask=masks[0], use_context=True),
            getattr(nn, self.act),
            MaskedDense(features=masks[1].shape[-1], mask=masks[1]),
            getattr(nn, self.act),
            MaskedDense(
                features=masks[2].shape[-1] * self.output_dim_multiplier,
                mask=masks[2].tile(self.output_dim_multiplier),
            ),
        ]

    def __call__(self, x, *args, **kwargs) -> Any:
        for layer in self.layers:
            if isinstance(layer, nn.Module):
                x = layer(x, *args, **kwargs)
            else:
                # activation functions don't take additional arguments
                x = layer(x)
        return x


class MADE(nn.Module):
    input_dim: int
    hidden_dim: int = 64
    context_dim: int = 0
    output_dim_multiplier: int = 2
    act: str = "celu"

    def setup(self):
        self.transform = MaskedTransform(
            input_dim=self.input_dim,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            num_hidden=1,
            output_dim_multiplier=self.output_dim_multiplier,
            act=self.act,
        )

    @compact
    def __call__(self, inputs, context=None):
        log_weight, bias = self.transform(inputs, context=context).split(
            self.output_dim_multiplier, axis=1
        )
        # print("forward", log_weight, bias)
        outputs = (inputs - bias) * np.exp(-log_weight)
        log_det_jacobian = -log_weight.sum(-1)
        return outputs, log_det_jacobian

    def forward(self, inputs, context=None):
        return self(inputs, context=context)

    def inverse(self, inputs, context=None):
        outputs = np.zeros_like(inputs)
        for i_col in range(inputs.shape[1]):
            log_weight, bias = self.transform(outputs, context=context).split(
                self.output_dim_multiplier, axis=1
            )
            # print("inverse", log_weight, bias)
            outputs = outputs.at[:, i_col].set(
                inputs[:, i_col] * np.exp(log_weight[:, i_col]) + bias[:, i_col]
            )
            
        log_det_jacobian = -log_weight.sum(-1)
        return outputs, log_det_jacobian


if __name__ == "__main__":
    import jax.numpy as np

    input_dim = 3
    hidden_dim = 64
    context_dim = 1
    output_dim_multiplier = 2

    rng = jax.random.PRNGKey(0)

    x = jax.numpy.ones((1, input_dim))
    context = jax.numpy.ones((1, context_dim))

    model = MADE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        context_dim=context_dim,
        output_dim_multiplier=2,
    )

    # test forward
    variables = model.init(rng, x, context)
    y = model.apply(variables, x, context)

    from IPython import embed

    embed()

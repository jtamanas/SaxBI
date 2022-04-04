import jax
import jax.numpy as np


def init_fn(rng, nn_fns, optimizer, input_shape, context_shape):
    if context_shape is None:
        context_shape = (0,)
    dummy_input = np.ones((1, *input_shape))
    dummy_context = np.ones((1, *context_shape))
    # shape inference done during init
    params = nn_fns.init(rng, dummy_input, dummy_context)
    opt_state = optimizer.init(params)
    return params, opt_state


parallel_init_fn = jax.vmap(
    init_fn, in_axes=(0, None, None, None, None), out_axes=(0, 0)
)

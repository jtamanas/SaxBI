import jax
import jax.numpy as np
import optax
from functools import partial
from .classifier import Classifier


def init_fn(input_shape, rng, classifier_fns, optimizer):
    dummy_input = np.ones((1, *input_shape))
    params = classifier_fns.init(rng, dummy_input)["params"]
    opt_state = optimizer.init(params)
    return params, opt_state


parallel_init_fn = jax.vmap(init_fn, in_axes=(None, 0, None, None))#, out_axes=(0, 0))


def InitializeClassifier(
    model_rng,
    optimizer,
    obs_dim,
    theta_dim,
    ensemble_size=5,
    num_layers=5,
    hidden_dim=128,
    **kwargs,
):
    """
    Initialize a likelihood ratio model.

    Args:
        model_rng: a jax random number generator
        obs_dim: dimensionality of the observations
        theta_dim: dimensionality of the simulation parameters
        num_layers: number of affine layers in the flow

    Returns:
        initial_params: a list of parameters
        log_pdf: a function from parameters to log-probability of the observations
        sample: a function from parameters to samples of the parameters

    """

    def loss(params, inputs, context, label):
        """binary cross entropy with logits
        taken from jaxchem
        """
        label = label.squeeze()
        # log ratio is the logit of the discriminator
        l_d = logit_d.apply({"params": params}, inputs, context).squeeze()
        max_val = np.clip(-l_d, 0, None)
        L = (
            l_d
            - l_d * label
            + max_val
            + np.log(np.exp(-max_val) + np.exp((-l_d - max_val)))
        )
        return np.mean(L)

    if type(model_rng) is int:
        model_rng = jax.random.PRNGKey(model_rng)
    ensemble_seeds = jax.random.split(model_rng, ensemble_size)

    logit_d = Classifier(num_layers=num_layers, hidden_dim=hidden_dim)

    initial_ensemble_params, initial_opt_state_ensemble = parallel_init_fn(
        (obs_dim + theta_dim,), ensemble_seeds, logit_d, optimizer
    )

    return loss, logit_d, initial_ensemble_params, initial_opt_state_ensemble

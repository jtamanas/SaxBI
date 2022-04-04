import jax
import jax.numpy as np
import flax.linen as nn

from saxbi.models.flows import flow, priors, utils, permutations, normalizations
import saxbi.models.flows.made as made_module


def get_loss_fn(flow_fns):
    def loss_fn(params, *args):
        """
        Negative-log-likelihood loss function.

        *args: inputs, context (optional)
        """
        # Do I have to make params into {"params": params}?
        return -flow_fns.apply(params, *args).mean()

    return loss_fn


def construct_MAF(
    rng: jax.random.PRNGKey,
    input_dim: int,
    hidden_dim: int = 64,
    context_dim: int = 0,
    n_layers: int = 5,
    context_embedding: nn.Module = None,
    permutation: str = "Reverse",
    normalization: str = None,
    made_activation: str = "celu",
):
    """
    A sequence of affine transformations with a masked affine transform.
    """

    if context_embedding is not None:
        context_dim = context_embedding.output_dim

    made_kwargs = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "context_dim": context_dim,
        "output_dim_multiplier": 2,
        "act": made_activation,
    }

    permutation = getattr(permutations, permutation)
    permutation_kwargs = {"input_dim": input_dim, "rng": None}

    if normalization is not None:
        normalization = getattr(normalizations, normalization)
    normalization_kwargs = {}

    transformations = []
    for rng in jax.random.split(rng, n_layers):
        permutation_kwargs["rng"] = rng

        transformations.append(made_module.MADE(**made_kwargs))
        transformations.append(permutation(**permutation_kwargs))
        if normalization is not None:
            transformations.append(normalization(**normalization_kwargs))

    maf = flow.Flow(
        transformation=utils.SeriesTransform(
            transformations=transformations,
            context_embedding=context_embedding,
        ),
        prior=priors.Normal(dim=input_dim),
    )

    return maf, get_loss_fn(maf)

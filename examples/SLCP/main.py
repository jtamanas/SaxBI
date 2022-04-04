import jax
import jax.numpy as np
import numpy as onp
from saxbi import pipeline

from saxbi.sampler import hmc
from simulator import get_simulator, X_true, theta_dim, log_prior, sample_prior

import corner
import matplotlib.pyplot as plt

# --------------------------

seed = 1234
rng, model_rng, hmc_rng = jax.random.split(jax.random.PRNGKey(seed), num=3)


model, params, Theta_post = pipeline(
    rng,
    X_true,
    get_simulator,
    # Prior
    log_prior,
    sample_prior,
    # Simulator
    simulator_kwargs={},
    # Model hyperparameters
    model_type="classifier",  # "classifier" or "flow"
    ensemble_size=15,
    num_layers=2,
    hidden_dim=32,
    # Optimizer hyperparmeters
    max_norm=1e-3,
    learning_rate=3e-4,
    weight_decay=1e-1,
    # Train hyperparameters
    nsteps=250000,
    patience=15,
    eval_interval=100,
    # Dataloader hyperparameters
    batch_size=32,
    train_split=0.8,
    num_workers=0,
    add_noise=False,
    # Sequential hyperparameters
    num_rounds=3,
    num_initial_samples=10000,
    num_samples_per_round_per_chain=1000,
    num_chains=10,
    logger=None,
)

parallel_log_prob = jax.vmap(model.apply, in_axes=(0, None, None))


def potential_fn(theta):
    if len(theta.shape) == 1:
        theta = theta[None, :]

    log_L = parallel_log_prob(params, X_true, theta)
    log_L = log_L.mean(axis=0)

    log_post = -log_L - log_prior(theta)
    return log_post.sum()


num_chains = 32
init_theta = sample_prior(rng, num_samples=num_chains)

mcmc = hmc(
    rng,
    potential_fn,
    init_theta,
    adapt_step_size=True,
    adapt_mass_matrix=True,
    dense_mass=True,
    step_size=1e0,
    max_tree_depth=6,
    num_warmup=2000,
    num_samples=2000,
    num_chains=num_chains,
)
mcmc.print_summary()

theta_samples = mcmc.get_samples(group_by_chain=False).squeeze()

theta_dim = theta_samples.shape[-1]
true_theta = onp.array([0.7, -2.9, -1.0, -0.9, 0.6])

corner.corner(
    onp.array(theta_samples),
    range=[(-3, 3) for i in range(theta_dim)],
    truths=true_theta,
    bins=75,
    smooth=(1.0),
    smooth1d=(1.0),
)

plt.show()


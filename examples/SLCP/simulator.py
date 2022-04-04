import jax
import jax.numpy as np
from saxbi.prior import SmoothedBoxPrior


def _calc_vars(theta: np.array):
    """
    Calculate the mean and variance of the posterior.


    mu = theta(1, 5) * mask(5, 2) = (1, 2) vector
    s_1, s_2 = theta(n, 5) * mask(5, 2) * theta(5, n) = (n,) scalar
    rho = theta(1, 5) * mask(5, 1) = (1, 1) vector

    Sigma = [[s_1 ** 2, rho * s_1 * s_2],
            [rho * s_1 * s_2, s_2 ** 2]]

    Sigma = outer([s_1, s_2], [s_1, s_2]) * elementwise_mult rho
    """
    # have to do it this way to allow vectorization
    s_vec = np.square(theta[2:4])
    rho = np.tanh(theta[5])
    rho_matrix = np.eye(2) + rho * np.eye(2)[::-1]  # off-diagonal rho

    mu = theta[:2]
    Sigma = np.outer(s_vec, s_vec) * rho_matrix
    return mu, Sigma


def log_likelihood(x: np.array, theta: np.array):
    """
    Calculate the log likelihood of the data given the posterior.
    """
    assert theta.ndim == 1 and len(theta) == 5, "theta must be a 1D array of length 5"
    mu, Sigma = _calc_vars(theta)

    x = x.reshape((4, 2))
    return jax.scipy.stats.multivariate_normal.logpdf(x, mean=mu, cov=Sigma).sum()


# sample from multi-variate gaussian
def sample(rng, theta: np.array, num_samples_per_theta: int):
    """
    Sample from the posterior.
    """

    def _sample(_th):
        mu, Sigma = _calc_vars(_th)
        samp = jax.random.multivariate_normal(
            rng, mu, Sigma, shape=(4 * num_samples_per_theta, 1)
        )
        samp = np.reshape(samp, (num_samples_per_theta, 4, -1))
        return samp

    assert theta.shape[-1], "theta must be a 1/2D array with 5D final dim"
    if theta.ndim == 1:
        theta = theta.reshape((1, theta.shape[0]))
    return np.reshape(
        jax.vmap(_sample)(theta), (theta.shape[0] * num_samples_per_theta, -1)
    )


def get_simulator():
    obs_dim = 8  # (4*2)
    theta_dim = 5
    simulate = sample
    return simulate, obs_dim, theta_dim


# --------------------------
# set up X_true 

simulator_kwargs = {}
true_theta = np.array([0.7, -2.9, -1.0, -0.9, 0.6])
simulate, obs_dim, theta_dim = get_simulator(**simulator_kwargs)
X_true = simulate(jax.random.PRNGKey(42), true_theta, num_samples_per_theta=1)


# --------------------------
# set up prior
log_prior, sample_prior = SmoothedBoxPrior(
    theta_dim=theta_dim, lower=-3.0, upper=3.0, sigma=0.02
)

if __name__ == "__main__":
    import numpy as onp
    from saxbi.prior import SmoothedBoxPrior
    import numpyro
    import corner
    import matplotlib.pyplot as plt
    import itertools

    seed = 123456789
    theta_dim = 5
    rng, hmc_rng = jax.random.split(jax.random.PRNGKey(seed), num=2)

    true_theta = np.array([0.7, -2.9, -1.0, -0.9, 0.6])
    observation = sample(rng, true_theta, num_samples_per_theta=1)

    lst = np.array(list(itertools.product([0, 1], repeat=theta_dim)))
    lst = (-1) ** lst
    lst = lst
    num_chains = len(lst)

    init_theta = [true_theta * row for i, row in enumerate(lst)]
    init_theta = np.stack(init_theta)
    init_theta = init_theta + 0.1 * jax.random.normal(rng, init_theta.shape)

    print("Sample from the posterior w/ hamiltonian monte carlo")

    log_prior, sample_prior = SmoothedBoxPrior(lower=-3.0, upper=3.0, sigma=0.01)

    def wrapper_log_posterior(theta):
        log_post = log_likelihood(observation, theta) + log_prior(theta).sum()
        return -log_post

    nuts_kernel = numpyro.infer.NUTS(
        potential_fn=wrapper_log_posterior,
    )
    mcmc = numpyro.infer.MCMC(
        nuts_kernel,
        num_samples=500,
        num_warmup=500,
        num_chains=num_chains,
    )

    mcmc.run(hmc_rng, init_params=init_theta)
    mcmc.print_summary()
    corner.corner(
        onp.array(mcmc.get_samples()),
        truths=true_theta,
        bins=75,
        range=[(-3, 3) for i in range(init_theta.shape[-1])],
        smooth=(1.0),
        smooth1d=(1.0),
    )
    plt.show()

import jax
import jax.numpy as np
from saxbi.prior import SmoothedBoxPrior


# sample from two moons
def sample(rng, theta: np.array, num_samples_per_theta: int):
    """
    Sample from the posterior.
    """

    def _sample(rng, _th):
        theta_1, theta_2 = _th 

        alpha = (jax.random.uniform(rng, shape=(1,)) - 0.5) * np.pi
        rng, _ = jax.random.split(rng, 2)
        r = jax.random.normal(rng, shape=(1 * num_samples_per_theta, 1)) * 0.01 + 0.1
        
        samp = np.array([r * np.sin(alpha) + 0.25 - np.abs(theta_1 + theta_2)/np.sqrt(2), r * np.sin(alpha) + (-theta_1 + theta_2)/np.sqrt(2)])
        samp = np.reshape(samp, (num_samples_per_theta, 2, -1))
        return samp

    if theta.ndim == 1:
        theta = theta.reshape((1, theta.shape[0]))
    return np.reshape(
        jax.vmap(_sample, in_axes=(None, 0))(rng, theta), (theta.shape[0] * num_samples_per_theta, -1)
    )


def get_simulator():
    obs_dim = 2  
    theta_dim = 2
    simulate = sample
    return simulate, obs_dim, theta_dim


# --------------------------
# set up X_true 

simulator_kwargs = {}
true_theta = np.array([-0.7, 0.8])
simulate, obs_dim, theta_dim = get_simulator(**simulator_kwargs)
X_true = simulate(jax.random.PRNGKey(42), true_theta, num_samples_per_theta=1)

# --------------------------
# set up prior
log_prior, sample_prior = SmoothedBoxPrior(
    theta_dim=theta_dim, lower=-1.0, upper=1.0, sigma=0.02
)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    rng = jax.random.PRNGKey(42)
    theta = sample_prior(rng, 1000)
    rng, _ = jax.random.split(rng, 2)
    
    samples = simulate(rng, theta, num_samples_per_theta=1)
    
    print(samples.shape)
import jax
import jax.numpy as np


def SmoothedBoxPrior(theta_dim=5, lower=0.0, upper=1.0, sigma=0.1, variance=False):
    assert np.all(lower < upper), "lower must be less than upper"
    assert np.all(sigma > 0), "sigma must be greater than zero"
    assert np.logical_xor(sigma, variance), "specify only one of sigma and variance"

    if not variance:
        variance = sigma ** 2
    _center = (upper + lower) / 2.0
    _range = (upper - lower) / 2.0

    def log_prob(theta):
        """Inspired by SmoothedBoxPrior From GPyTorch

        If theta is inside the bounds, return constant.
        If theta is outside the bounds, return log prob from sharp normal

        Can accomplish this saying the distance from the edges of the theta range
        is sampled from a normal distribution (clipped at zero to not go negative)
        """
        _theta_dist = np.clip(np.abs(theta - _center) - _range, 0, None)
        return -0.5 * (_theta_dist ** 2 / variance + np.log(2 * np.pi * variance))

    def sample(rng, num_samples: int = 1):
        """
        Samples are taken from a hard uniform distribution between the bounds
        """
        return jax.random.uniform(
            rng, shape=(num_samples, theta_dim), minval=lower, maxval=upper
        )

    return log_prob, sample

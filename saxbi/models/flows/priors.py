import jax
from jax.scipy.stats import norm
import flax.linen as nn 


class Normal(nn.Module):
    dim: int
    
    def log_prob(self, inputs):
        return norm.logpdf(inputs).sum(1)

    def sample(self, rng, num_samples=1):
        return jax.random.normal(rng, (num_samples, self.dim))

    def sample_with_log_prob(self, rng, num_samples=1):
        samples = self.sample(rng, num_samples)
        log_probs = self.log_prob(samples)
        return samples, log_probs
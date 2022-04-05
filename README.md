<div align="center">
<img src="logo.png" alt="logo"  width="250"><img\>
</div>

SaxBI is a JAX implementation of likelihood-free simulation-based inference (sbi) methods. Currently, the two algorithms used are Sequential Neural Likelihood Estimation (SNLE) and Sequential Neural Ratio Estimation (SNRE). This package offers a simple, functional API for carrying out the approximate posterior inference.
 
 
### The fully automated pipeline features:

 * [Flax](https://github.com/google/flax)-based autoregressive normalizing flows with affine, piecewise affine, and piecewise rational quadratic splines
 * [Flax](https://github.com/google/flax)-based classifiers with/out residual skip connections
 * Hamiltonian Monte Carlo sampling with NUTS kernels implemented in [Numpyro](https://github.com/pyro-ppl/numpyro)
 * And more! 
 * Probably some bugs too... Let me know what you find 😅
 
# Installation

`saxbi` requires python 3.9 or higher. It can be easily installed from the repository's home directory with 

```
python setup.py install
```



# Basic Usage

The main workhorse of this package is the `pipeline` function which takes 5 required arguments: `rng`, `X_true`, `get_simulator`, `log_prior`, and `sample_prior`. We recommend making a `simulator.py` file from which the latter 4 of these can be imported. The `pipeline` function then returns the flax model, its trained parameters, and samples from the final iteration of the posterior.

```python
from saxbi import pipeline
from simulator import X_true, get_simulator, log_prior, sample_prior

rng = jax.random.PRNGKey(16)

model, params, Theta_post = pipeline(rng, X_true, get_simulator, log_prior, sample_prior)
```


The [`examples/`](https://github.com/jtamanas/LBI/tree/master/SaxBI/examples) directory holds a few canonical examples from the literature to show off the syntax in greater detail. 


# SBI Algorithm References

#### Sequential Neural Likelihood Estimation (SNLE)
* `SNLE` from Papamakarios G, Sterrat DC and Murray I [_Sequential
  Neural Likelihood_](https://arxiv.org/abs/1805.07226).

#### Sequential Neural likelihood-to-evidence Ratio Estimation (SNRE)

* `SNRE` from Hermans J, Begy V, and Louppe G. [_Likelihood-free Inference with Amortized Approximate Likelihood Ratios_](https://arxiv.org/abs/1903.04057).


# Todo
* Add diagnostics (like MMD, ROC AUC)
* Add support for [Mining Gold](https://arxiv.org/abs/1805.12244) (i.e. using simulator derivatives to improve likelihood estimators)

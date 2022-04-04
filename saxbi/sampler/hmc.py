import numpyro


def hmc(
    hmc_rng,
    potential_fn,
    init_theta,
    adapt_step_size=True,
    adapt_mass_matrix=True,
    dense_mass=True,
    step_size=1e0,
    max_tree_depth=12,
    num_warmup=100,
    num_samples=50,
    num_chains=1,
    extra_fields=(),
    chain_method="vectorized",
):

    nuts_kernel = numpyro.infer.NUTS(
        potential_fn=potential_fn,
        adapt_step_size=adapt_step_size,
        adapt_mass_matrix=adapt_mass_matrix,
        dense_mass=dense_mass,
        step_size=step_size,
        max_tree_depth=max_tree_depth,
    )
    mcmc = numpyro.infer.MCMC(
        nuts_kernel,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
        chain_method=chain_method,
    )

    # TODO: make sure x0 is being used in posterior (because it's not being used in run)
    mcmc.run(hmc_rng, init_params=init_theta, extra_fields=extra_fields)
    return mcmc

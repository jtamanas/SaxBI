import jax
import jax.numpy as np
import numpy as onp
from ..sampler.hmc import hmc


"""
1. Sample theta from prior
2. Simulate X from theta
3. Assemble dataset
4. Train model (flow or classifier should'nt matter)
5. Repeat at 1 but replace prior with posterior
"""


def _sample_theta(rng, sample, num_theta):
    """
    Sample theta from theta_prior

    rng: jax.random.PRNGKey
    sample: function that samples from theta_prior
    num_samples: number of samples to draw
    """
    return sample(rng, num_theta)


def _simulate_X(rng, simulate, theta, num_samples_per_theta: int = 1):
    """
    Simulate X from theta

    simulator: simulator function
    theta: theta
    num_samples_per_theta: number of samples to draw from simulator per theta
    """
    return simulate(rng, theta, num_samples_per_theta=num_samples_per_theta)


def _train_model(
    trainer, model_params, opt_state, train_dataloader, valid_dataloader, num_round=0
):
    """
    Train model
    """
    return trainer(
        model_params, opt_state, train_dataloader, valid_dataloader, num_round=num_round
    )


def _sample_posterior(
    rng,
    model_params,
    log_prob,
    log_prior,
    X_true,
    init_theta=None,
    num_samples=1000,
    num_chains=32,
    num_warmup=None,
):
    def potential_fn(theta):
        if len(theta.shape) == 1:
            theta = theta[None, :]

        log_L = log_prob(model_params, X_true, theta)
        log_L = log_L.mean(axis=0)

        log_post = -log_L - log_prior(theta)
        return log_post.sum()

    if num_warmup is None:
        num_warmup = num_samples

    mcmc = hmc(
        rng,
        potential_fn,
        init_theta,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        dense_mass=True,
        step_size=1e0,
        max_tree_depth=6,
        num_warmup=num_samples,
        num_samples=num_samples,
        num_chains=num_chains,
    )
    mcmc.print_summary()

    return mcmc.get_samples(group_by_chain=False).squeeze()


def _get_init_theta(model_params, log_prob, X_true, Theta, num_theta=1):
    tiled_X = np.tile(X_true, (Theta.shape[0], 1))
    lps = log_prob(model_params, tiled_X, Theta).squeeze()
    init_theta = np.array(Theta[np.argsort(lps)])[:num_theta]
    return init_theta


def _sequential_round(
    rng,
    X_true,
    model_params,
    log_prob,
    log_prior,
    sample_prior,
    simulate,
    opt_state,
    trainer,
    data_loader_builder,
    get_init_theta,
    num_initial_samples=1000,
    num_samples_per_round_per_chain=100,
    num_samples_per_theta=1,
    num_chains=32,
    num_warmup_per_round=None,
    Theta_Old=None,
    Theta_New=None,
    X=None,
    num_round=0,
):
    # TODO: These if statements are a bit of a mess. Should be refactored.
    if num_warmup_per_round is None:
        num_warmup_per_round = num_samples_per_round_per_chain
    if Theta_New is not None:
        X_New = _simulate_X(rng, simulate, Theta_New, num_samples_per_theta)

        # Remove nans
        nan_idx = onp.any(onp.isnan(X_New), axis=1)
        X_New = X_New[~nan_idx]
        Theta_New = Theta_New[~nan_idx]
        
        print(f'Number of non-nan samples: {X_New.shape[0]}')

    if Theta_Old is not None:
        Theta = np.vstack([Theta_Old, Theta_New])
    else:
        Theta = Theta_New

    if X is not None:
        X = np.vstack([X, X_New])
    else:
        X = X_New


    train_dataloader, valid_dataloader = data_loader_builder(X=X, Theta=Theta)
    model_params = _train_model(
        trainer,
        model_params,
        opt_state,
        train_dataloader,
        valid_dataloader,
        num_round=num_round,
    )

    # some mcmc stuff

    init_theta = get_init_theta(
        model_params.slow if hasattr(model_params, "slow") else model_params,
        log_prob,
        X_true,
        Theta,
        num_theta=num_chains,
    )

    Theta_post = _sample_posterior(
        rng,
        model_params.slow if hasattr(model_params, "slow") else model_params,
        log_prob,
        log_prior,
        X_true,
        init_theta=init_theta,
        num_samples=num_samples_per_round_per_chain,
        num_warmup=num_samples_per_round_per_chain,
        num_chains=num_chains,
    )
    return model_params, X, Theta, Theta_post


def sequential(
    rng,
    X_true,
    model_params,
    log_prob,
    log_prior,
    sample_prior,
    simulate,
    opt_state,
    trainer,
    data_loader_builder,
    get_init_theta=None,
    num_rounds=10,
    num_initial_samples=1000,
    num_samples_per_round_per_chain=100,
    num_samples_per_theta=1,
    num_chains=32,
    num_warmup_per_round=None,
    Theta=None,
    X=None,
    normalize=None,
    logger=None,
):
    if num_warmup_per_round is None:
        num_warmup_per_round = num_samples_per_round_per_chain
    if get_init_theta is None:
        # get_init_theta = _get_init_theta
        get_init_theta = lambda mp, lp, xt, th, num_theta=num_chains: sample_prior(
            rng, num_samples=num_theta
        )

    if Theta is None:
        Theta_New = _sample_theta(rng, sample_prior, num_initial_samples)
    else:
        Theta_New = Theta
        
        
    for i in range(num_rounds):
        print(f"STARTING ROUND {i+1}")
        model_params, X, Theta, Theta_New = _sequential_round(
            rng,
            X_true,
            model_params,
            log_prob,
            log_prior,
            sample_prior,
            simulate,
            opt_state,
            trainer,
            data_loader_builder,
            get_init_theta,
            num_initial_samples=num_initial_samples,
            num_samples_per_round_per_chain=num_samples_per_round_per_chain,
            num_samples_per_theta=num_samples_per_theta,
            num_chains=num_chains,
            Theta_Old=Theta,
            Theta_New=Theta_New,
            X=X,
            num_round=i,
        )

        # DEBUGGING
        # TODO: Add diagnostics to each round
        import corner
        import matplotlib.pyplot as plt
        import numpy as onp
        from saxbi.diagnostics import AUC, MMD

        theta_dim = Theta.shape[-1]

        # try:
        #     corner.corner(
        #         onp.array(Theta_New),
        #         range=[(0, 1) for i in range(theta_dim)],
        #         bins=75,
        #         smooth=(1.0),
        #         smooth1d=(1.0),
        #     )

        #     if hasattr(logger, "plot"):
        #         logger.plot(f"diagnostic_corner", plt, close_plot=True, step=(i+1))
        #     else:
        #         plt.show()
        # except KeyboardInterrupt:
        #     pass
        # DEBUGGING

    return model_params, Theta_New

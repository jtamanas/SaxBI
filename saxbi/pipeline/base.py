import jax
import jax.numpy as np
import numpy as onp
import optax
from saxbi.dataset import getDataLoaderBuilder
from saxbi.sequential.sequential import sequential
from saxbi.models import parallel_init_fn
from saxbi.models.steps import get_train_step, get_valid_step
from saxbi.models.flows import construct_MAF
from saxbi.models.MLP import MLP
from saxbi.models.classifier import construct_Classifier
from saxbi.models.classifier.classifier import get_loss_fn
from saxbi.trainer import getTrainer


def pipeline(
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
    sigma=None,
    add_noise=False,
    scale_X=None,
    inverse_scale_X=None,
    scale_Theta=None,
    # Sequential hyperparameters
    num_rounds=3,
    num_initial_samples=1000,
    num_warmup_per_round=100,
    num_samples_per_round_per_chain=100,
    num_chains=10,
    logger=None,
):
    """
    
    Parameters:
    -----------
    rng: jax.random.PRNGKey

    X_true: np.ndarray
        The true data vector on which the posterior is conditioned.

    get_simulator: function
        Takes in simulator_kwargs and returns the simulator function, obs_dim, and theta_dim
            (where obs_dim is the dimension of the observed data, and 
            theta_dim is the dimension of the parameter vector)

    log_prior: function
        Takes in theta and returns the log prior probability of theta.

    sample_prior: function
        Takes in rng and returns a sample from the prior.

    simulator_kwargs: dict
        The keyword arguments to pass to the simulator function.

    model_type: str
        The type of model to use. Can be either "classifier" or "flow".

    ensemble_size: int
        The number of density estimation models to use in the ensemble.

    num_layers: int
        The number of layers in the MLP OR the number of transforms in the MAF.

    hidden_dim: int
        The number of hidden units in each layer of the MLP 
            OR the number of hidden units in each MAF transform.

    max_norm: float
        The maximum norm of the gradient.

    learning_rate: float
        The learning rate for the Adam optimizer.

    weight_decay: float
        The weight decay for the Adam optimizer.

    nsteps: int
        The maximum number of steps to take during train.

    patience: int
        The number of evaluation intervals to wait before early stopping.

    eval_interval: int 
        The number of training steps to wait before evaluating the model.

    batch_size: int
        The number of samples to use in each batch during training.

    train_split: float
        The fraction of the dataset to use for training. The rest is used for validation.

    num_workers: int
        The number of workers to use for the dataloader.

    scale_X: function
        Takes in X and returns the scaled version of X.  
            Passed into dataloader builder to scale the data.

    scale_Theta: function
        Takes in Theta and returns the scaled version of Theta. 
            Passed into dataloader builder to scale theta.

    num_rounds: int
        The number of rounds to run the sequential algorithm. num_rounds = 1 is equivalent to the amortized, non-sequential algorithm.

    num_initial_samples: int
        The number of samples to use to in the first round of the algorithm.

    num_warmup_per_round: int
        The number of warmup steps to use in each round of sampling from the posterior with HMC.

    num_samples_per_round_per_chain: int
        The number of samples to obtain from each chain in each round of sampling from the posterior with HMC.

    num_chains: int
        The number of chains to use in each round of sampling from the posterior with HMC.

    logger: trax.jaxboard.SummarWriter or None
    """

    # --------------------------
    # set up simulation and observables
    simulate, obs_dim, theta_dim = get_simulator(**simulator_kwargs)

    data_loader_builder = getDataLoaderBuilder(
        sequential_mode=model_type,
        batch_size=batch_size,
        train_split=train_split,
        num_workers=num_workers,
        sigma=sigma,
        add_noise=add_noise,
        scale_X=scale_X,
        inverse_scale_X=inverse_scale_X,
        scale_Theta=scale_Theta,
    )

    # --------------------------
    # Create optimizer
    optimizer = optax.chain(
        # Set the parameters of Adam optimizer
        optax.adamw(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
        ),
        optax.adaptive_grad_clip(max_norm),
    )

    # --------------------------
    # Create model
    if model_type == "classifier":

        classifier_kwargs = {
            # "output_dim": 1,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "use_residual": False,
            "act": "gelu",
        }
        model, loss_fn = construct_Classifier(**classifier_kwargs)
    else:
        maf_kwargs = {
            "rng": rng,
            "input_dim": obs_dim,
            "hidden_dim": hidden_dim,
            "context_dim": theta_dim,
            "n_layers": num_layers,
            "permutation": "Conv1x1",
            "normalization": None,
            "made_activation": "gelu",
        }
        context_embedding_kwargs = {
            "output_dim": theta_dim * 2,
            "hidden_dim": theta_dim * 2,
            "num_layers": 2,
            "act": "leaky_relu",
        }

        context_embedding = MLP(**context_embedding_kwargs)
        model, loss_fn = construct_MAF(
            context_embedding=context_embedding, **maf_kwargs
        )

    params, opt_state = parallel_init_fn(
        jax.random.split(rng, ensemble_size),
        model,
        optimizer,
        (obs_dim,),
        (theta_dim,),
    )

    # the models' __call__ are their log_prob fns
    parallel_log_prob = jax.vmap(model.apply, in_axes=(0, None, None))
    # --------------------------
    # Create trainer

    train_step = get_train_step(loss_fn, optimizer)
    valid_step = get_valid_step({"valid_loss": loss_fn})

    trainer = getTrainer(
        train_step,
        valid_step=valid_step,
        nsteps=nsteps,
        eval_interval=eval_interval,
        patience=patience,
        logger=logger,
        train_kwargs=None,
        valid_kwargs=None,
    )

    # Train model sequentially
    params, Theta_post = sequential(
        rng,
        X_true,
        params,
        parallel_log_prob,
        log_prior,
        sample_prior,
        simulate,
        opt_state,
        trainer,
        data_loader_builder,
        num_rounds=num_rounds,
        num_initial_samples=num_initial_samples,
        num_samples_per_round_per_chain=num_samples_per_round_per_chain,
        num_samples_per_theta=1,
        num_chains=num_chains,
        logger=logger,
    )

    return model, params, Theta_post

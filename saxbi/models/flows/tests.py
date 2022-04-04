from functools import partial
import jax
import jax.numpy as np
from saxbi.models.flows import construct_MAF


def check_invertibility(seed=42):
    """
    Check if flow is invertible.
    """
    input_dim = 5
    context_dim = 10
    hidden_dim = 64
    n_layers = 5
    rng = jax.random.PRNGKey(seed)

    sample_input = 10 * (jax.random.uniform(rng, (1, input_dim)) - 0.5)
    sample_context = 10 * (jax.random.uniform(rng, (1, context_dim)) - 0.5)

    def init_fn(rng, input_shape, context_shape=None):
        if context_shape is None:
            context_shape = (0,)
        dummy_input = np.ones((1, *input_shape))
        dummy_context = np.ones((1, *context_shape))
        params = maf.init(rng, dummy_input, context=dummy_context)
        return params

    maf = construct_MAF(
        rng=rng,
        input_dim=input_dim,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        context_embedding=None,
    )

    params = init_fn(
        rng=rng,
        input_shape=(input_dim,),
        context_shape=(context_dim,),
    )

    sample_latent = maf.apply(params, sample_input, sample_context, method=maf.forward)[
        0
    ]
    sample_input_back = maf.apply(
        params, sample_latent, sample_context, method=maf.inverse
    )[0]

    delta = sample_input - sample_input_back
    print("delta", delta)

    if np.allclose(sample_input, sample_input_back):
        print("Flow is invertible")
    else:
        print("Flow is not invertible")
        print("delta std:", np.std(delta))

    print("______________________________________________________")
    print("_____________ Conditonal Two  Moons Test _____________")
    print("______________________________________________________")


def two_moons(seed=42):
    import jax
    import functools
    from saxbi.models import get_train_step, get_valid_step, init_fn, parallel_init_fn
    from saxbi.models.MLP import MLP
    import optax
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt

    rng = jax.random.PRNGKey(seed)

    learning_rate = 1e-3
    batch_size = 128
    nsteps = 40

    n_layers = 1
    hidden_dim = 32
    ensemble_size = 5 
    # speed of training ensemble depends on the size of:
    # ensemble, batch, and hidden dim

    # --------------------
    # Create the dataset
    # --------------------
    X, y = make_moons(n_samples=10000, noise=0.05, random_state=seed)
    y = y[:, None]
    input_dim = X.shape[1]
    context_dim = y.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, stratify=y, random_state=seed
    )

    X_train_s = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    train_dataloader = DataLoader(
        TensorDataset(X_train_s, y_train), batch_size=batch_size, shuffle=True
    )

    # --------------------
    # Create the model
    # --------------------

    maf_kwargs = {
        "rng": rng,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "context_dim": context_dim,
        "n_layers": n_layers,
        "permutation": "Reverse",
        "normalization": None,
        "made_activation": "gelu",
    }

    context_embedding_kwargs = {
        "output_dim": 4,
        "hidden_dim": 8,
        "num_layers": 1,
        "act": "leaky_relu",
    }

    context_embedding = MLP(**context_embedding_kwargs)
    maf, loss_fn = construct_MAF(context_embedding=context_embedding, **maf_kwargs)

    optimizer = optax.adam(learning_rate=learning_rate)

    params, opt_state = parallel_init_fn(
        jax.random.split(rng, ensemble_size),
        maf,
        optimizer,
        (input_dim,),
        (context_dim,),
    )

    train_step = get_train_step(loss_fn, optimizer)

    iterator = tqdm(range(nsteps))
    try:
        for _ in iterator:
            for batch in train_dataloader:
                batch = [np.array(a) for a in batch]
                nll, params, opt_state = train_step(params, opt_state, batch)
            iterator.set_description("nll = {:.3f}".format(nll.mean()))
    except KeyboardInterrupt:
        pass

    plt.scatter(*X_train.T, color="grey", alpha=0.01, marker=".")

    # This is parallel sampling bit is ugly....
    # Technically not correct bc samples should be rejected 
    # rather than split evenly. As n_samp->inf this is fine.

    sample_0 = functools.partial(
        maf.apply,
        context=np.zeros((1000 // ensemble_size, context_dim)),
        method=maf.sample,
    )
    sample_1 = functools.partial(
        maf.apply,
        context=np.ones((1000 // ensemble_size, context_dim)),
        method=maf.sample,
    )
    parallel_sample_0 = jax.vmap(sample_0, in_axes=(0, 0))
    parallel_sample_1 = jax.vmap(sample_1, in_axes=(0, 0))
    
    samples_0 = parallel_sample_0(
        params,
        jax.random.split(rng, ensemble_size),
    )
    samples_1 = parallel_sample_1(
        params,
        jax.random.split(rng, ensemble_size),
    )

    plt.scatter(*samples_0.T, color="red", alpha=0.1, marker=".", label="0")
    plt.scatter(*samples_1.T, color="blue", alpha=0.1, marker=".", label="1")

    plt.xlim(-1.5, 2.5)
    plt.ylim(-1, 1.5)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # check_invertibility()
    two_moons()

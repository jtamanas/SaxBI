import jax.numpy as np


# ADAPTED FROM https://github.com/pratikm141/MMD-Variational-Autoencoder-Pytorch-InfoVAE/
def compute_kernel(x, y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]

    tiled_x = np.expand_dims(x, 1).tile((1, y_size, 1))
    tiled_y = np.expand_dims(y, 0).tile((x_size, 1, 1))

    return np.exp(-np.mean((tiled_x - tiled_y) ** 2, axis=2) / dim * 1.0)


def MMD(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return np.mean(x_kernel) + np.mean(y_kernel) - 2 * np.mean(xy_kernel)


if __name__ == "__main__":
    import jax

    seed = 1234
    rng_key = jax.random.PRNGKey(seed)

    num_data = 1000
    data_dim = 5

    for offset in np.linspace(0, 3, 10):

        mean = np.zeros(data_dim)
        cov = np.eye(data_dim)
        a_samples = jax.random.multivariate_normal(
            rng_key, mean=mean, cov=cov, shape=(num_data,)
        )
        b_samples = jax.random.multivariate_normal(
            rng_key, mean=mean + offset, cov=cov, shape=(num_data,)
        )

        print(f"Offset: {offset:.1f}", MMD(a_samples, b_samples))

import jax
import jax.numpy as np

import flax.linen as nn


def get_loss_fn(classifier_fns):
    def binary_cross_entropy(params, inputs, context, label):
        """binary cross entropy with logits
        taken from jaxchem
        """
        label = label.squeeze()
        # log ratio is the logit of the discriminator
        l_d = classifier_fns.apply(params, inputs, context).squeeze()
        max_val = np.clip(-l_d, 0, None)
        L = (
            l_d
            - l_d * label
            + max_val
            + np.log(np.exp(-max_val) + np.exp((-l_d - max_val)))
        )
        return np.mean(L)

    return binary_cross_entropy


class ResidualBlock(nn.Module):
    hidden_dim: int
    act: str = "celu"

    @nn.compact
    def __call__(self, x):
        y = nn.Dense(self.hidden_dim)(x)
        y = getattr(nn, self.act)(y)
        y = nn.Dense(x.shape[-1])(y)
        return x + y


class Classifier(nn.Module):
    # TODO: replace with MLP
    num_layers: 5
    hidden_dim: 128
    use_residual: bool = False
    act: str = "celu"

    @nn.compact
    def __call__(self, *args):
        x = np.hstack(args)
        for layer in range(self.num_layers):
            if self.use_residual:
                x = ResidualBlock(self.hidden_dim, self.act)(x)
            else:
                x = nn.Dense(self.hidden_dim)(x)
                x = getattr(nn, self.act)(x)
        return nn.Dense(1)(x)

    def log_prob(self, *args):
        return self(self, *args)


def construct_Classifier(num_layers=5, hidden_dim=128, use_residual=False, act="celu"):
    model = Classifier(
        num_layers=num_layers, hidden_dim=hidden_dim, use_residual=use_residual, act=act
    )

    return model, get_loss_fn(model)


if __name__ == "__main__":
    import optax
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from functools import partial
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    from tqdm.auto import tqdm

    def loss(params, inputs, targets, eps=1e-4):
        """binary cross entropy with logits
        taken from jaxchem
        """
        # log ratio is the logit of the discriminator
        l_d = logit_d.apply({"params": params}, inputs).squeeze()
        max_val = np.clip(-l_d, 0, None)
        L = (
            l_d
            - l_d * targets
            + max_val
            + np.log(np.exp(-max_val) + np.exp((-l_d - max_val)))
        )

        return np.sum(L)

    def get_train_step(loss, optimizer):
        @jax.jit
        def step(params, opt_state, batch):
            inputs, labels = batch
            nll, grads = jax.value_and_grad(loss)(params, inputs, labels)
            updates, opt_state = optimizer.update(grads, opt_state, params)

            return nll, optax.apply_updates(params, updates), opt_state

        return step

    nsteps = 50
    batch_size = 64
    seed = 1234

    X, y = load_breast_cancer(return_X_y=True)
    num_feat = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed
    )

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_train_s = torch.tensor(X_train_s, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    train_dataloader = DataLoader(
        TensorDataset(X_train_s, y_train), batch_size=batch_size, shuffle=True
    )

    learning_rate = 0.001
    optimizer = optax.chain(
        # Set the parameters of Adam. Note the learning_rate is not here.
        optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
        # Put a minus sign to *minimise* the loss.
        optax.scale(-learning_rate),
    )

    logit_d = Classifier(num_layers=3, hidden_dim=128, use_residual=True)
    params, opt_state = init_fn(
        X_train_s.shape, jax.random.PRNGKey(seed), logit_d, optimizer
    )

    train_step = get_train_step(loss, optimizer)

    print(
        "Test accuracy: {:.3f}".format(
            jax.numpy.mean(
                (
                    jax.nn.sigmoid(
                        logit_d.apply({"params": params}, np.array(X_test_s))
                    )
                    > 0.5
                ).flatten()
                == y_test
            )
        )
    )

    iterator = tqdm(range(nsteps))
    for _ in iterator:
        for batch in train_dataloader:
            batch = [np.array(a) for a in batch]
            nll, params, opt_state = train_step(params, opt_state, batch)
        iterator.set_description("nll = {:.3f}".format(nll))

    print()
    print(
        "Test accuracy: {:.3f}".format(
            jax.numpy.mean(
                (
                    jax.nn.sigmoid(
                        logit_d.apply({"params": params}, np.array(X_test_s))
                    )
                    > 0.5
                ).flatten()
                == y_test
            )
        )
    )
    print(
        classification_report(
            y_test,
            (
                jax.nn.sigmoid(logit_d.apply({"params": params}, np.array(X_test_s)))
                > 0.5
            ).flatten(),
        )
    )

    # fun fact: the residual network has slightly better
    # performance than the vanilla network on the test set

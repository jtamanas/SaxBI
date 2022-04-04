import jax
import jax.numpy as np
import numpy as onp
import optax
import sklearn.metrics as skm
import torch
from saxbi.models import get_train_step, get_valid_step
# from saxbi.models.classifier import InitializeClassifier
from saxbi.trainer import getTrainer
from sklearn.model_selection import train_test_split


def data_loader_builder(a_samples, b_samples, train_split=0.9):

    train_a, valid_a, train_b, valid_b = train_test_split(
        a_samples, b_samples, train_size=train_split
    )

    train_data = np.vstack([train_a, train_b])
    train_labels = np.vstack(
        [np.zeros((train_a.shape[0], 1)), np.ones((train_b.shape[0], 1))]
    )

    valid_data = np.vstack([valid_a, valid_b])
    valid_labels = np.vstack(
        [np.zeros((valid_a.shape[0], 1)), np.ones((valid_b.shape[0], 1))]
    )

    train_data = torch.tensor(onp.array(train_data), dtype=torch.float32)
    train_labels = torch.tensor(onp.array(train_labels), dtype=torch.float32)
    valid_data = torch.tensor(onp.array(valid_data), dtype=torch.float32)
    valid_labels = torch.tensor(onp.array(valid_labels), dtype=torch.float32)

    DSet = torch.utils.data.TensorDataset

    trainDataset = DSet(train_data, train_labels)
    validDataset = DSet(valid_data, valid_labels)

    train_dataloader = torch.utils.data.DataLoader(
        trainDataset,
        batch_size=128,
        shuffle=True,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        validDataset,
        batch_size=128 * 16,
        shuffle=True,
    )

    return train_dataloader, valid_dataloader


def train_classifier(
    rng_key,
    a_samples,
    b_samples,
    train_split=0.9,
    learning_rate=1e-4,
    train_nsteps=10000,
    eval_interval=10,
    patience=5000,
    num_layers=3,
    hidden_dim=128,
):
    def loss(params, inputs, label):
        """binary cross entropy with logits
        taken from jaxchem

        default loss takes context, so we specify this one to circumvent that
        """
        label = label.squeeze()
        # log ratio is the logit of the discriminator
        l_d = logit_d(params, inputs).squeeze()
        max_val = np.clip(-l_d, 0, None)
        L = (
            l_d
            - l_d * label
            + max_val
            + np.log(np.exp(-max_val) + np.exp((-l_d - max_val)))
        )
        return np.mean(L)

    # --------------------------
    # Create model
    model_params, _, logit_d = InitializeClassifier(
        model_rng=rng_key,
        obs_dim=a_samples.shape[-1],
        theta_dim=0,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )

    # --------------------------
    # Create optimizer
    optimizer = optax.chain(
        # Set the parameters of Adam optimizer
        optax.adamw(
            learning_rate=learning_rate,
            weight_decay=1e-2,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
        ),
    )
    opt_state = optimizer.init(model_params)

    # --------------------------
    # Create trainer

    train_step = get_train_step(loss, optimizer)
    valid_step = get_valid_step({"valid_loss": loss, "also_valid_loss": loss})

    trainer = getTrainer(
        train_step,
        valid_step=valid_step,
        nsteps=train_nsteps,
        eval_interval=eval_interval,
        patience=patience,
        logger=None,
        train_kwargs=None,
        valid_kwargs=None,
    )

    train_dataloader, valid_dataloader = data_loader_builder(
        a_samples, b_samples, train_split=train_split
    )

    model_params = trainer(
        model_params,
        opt_state,
        train_dataloader,
        valid_dataloader=valid_dataloader,
    )

    return model_params, logit_d


def distinguish_samples(a_samples, b_samples, classifier_params, logit_d):
    a_labels = np.zeros((a_samples.shape[0], 1))
    b_labels = np.ones((b_samples.shape[0], 1))
    samples = np.vstack([a_samples, b_samples])
    true_labels = np.vstack([a_labels, b_labels])

    pred_labels = logit_d(classifier_params, samples)
    pred_labels = jax.nn.sigmoid(pred_labels)
    return true_labels, pred_labels


def ROC_AUC(
    rng_key,
    a_samples,
    b_samples,
    train_split=0.9,
    learning_rate=1e-4,
    train_nsteps=10000,
    eval_interval=10,
    patience=5000,
    num_layers=3,
    hidden_dim=128,
):
    classifier_params, logit_d = train_classifier(
        rng_key,
        a_samples,
        b_samples,
        train_split=train_split,
        learning_rate=learning_rate,
        train_nsteps=train_nsteps,
        eval_interval=eval_interval,
        patience=patience,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )

    true_labels, pred_labels = distinguish_samples(
        a_samples, b_samples, classifier_params, logit_d
    )

    fpr, tpr, _ = skm.roc_curve(true_labels, pred_labels)
    roc_auc = skm.auc(fpr, tpr)

    return fpr, tpr, roc_auc


def AUC(
    rng_key,
    a_samples,
    b_samples,
    train_split=0.9,
    learning_rate=1e-4,
    train_nsteps=10000,
    eval_interval=10,
    patience=5000,
    num_layers=3,
    hidden_dim=128,
):
    """
    Just a simple wrapper to get the AUC from ROC_AUC
    """
    _, _, roc_auc = ROC_AUC(
        rng_key,
        a_samples,
        b_samples,
        train_split=train_split,
        learning_rate=learning_rate,
        train_nsteps=train_nsteps,
        eval_interval=eval_interval,
        patience=patience,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )

    return roc_auc


def reweigh_samples(rng, model_params, log_r, data, context, data_split=0.1):
    """
    This is a function needed reweigh the samples from a dataset
    Specifically, p(x|theta) = p(x)*r(x|theta)
    so we take p(x) samples and then choose the ones that have the highest
    probability of being generated by theta.

    Note: data_split=1 means the returned samples are sampled from p(x)
    
    # TODO: Refactor this to be a standalone sampler for classifier
    
    """
    size = int(data.shape[0] * data_split)
    weights = np.exp(log_r(model_params, data, context))
    weights = weights.squeeze()
    weights = weights / np.sum(weights)
    idx = np.arange(data.shape[0])
    weighted_idx = jax.random.choice(rng, idx, p=weights, replace=False, shape=(size,))
    weighted_samples = data[weighted_idx]
    return data[:size], weighted_samples


def LR_ROC_AUC(
    rng,
    model_params,
    log_r,
    data,
    context,
    data_split=0.1,
    train_split=0.9,
    learning_rate=1e-4,
    train_nsteps=10000,
    eval_interval=10,
    patience=5000,
    num_layers=3,
    hidden_dim=128,
):
    """
    This is the wrapper needed to get the ROC for classifiers
    """
    a_samples, b_samples = reweigh_samples(
        rng, model_params, log_r, data, context, data_split=data_split
    )
    
    fpr, tpr, roc_auc = ROC_AUC(
        rng,
        a_samples,
        b_samples,
        train_split=train_split,
        learning_rate=learning_rate,
        train_nsteps=train_nsteps,
        eval_interval=eval_interval,
        patience=patience,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )
    return fpr, tpr, roc_auc


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def ROC_of_true_discriminator(rng_key, a_samples, b_samples):
        """
        The optimal discriminator is the ratio of d(x) = prob_a(x)/(prob_a(x) + prob_b(x))
        """
        dist_a = lambda x: jax.scipy.stats.multivariate_normal.pdf(
            x, mean=mean, cov=cov
        )
        dist_b = lambda x: jax.scipy.stats.multivariate_normal.pdf(
            x, mean=mean + offset, cov=cov
        )

        def opt_d(x):
            return dist_b(x) / (dist_a(x) + dist_b(x))

        a_labels = np.zeros((a_samples.shape[0], 1))
        b_labels = np.ones((b_samples.shape[0], 1))
        samples = np.vstack([a_samples, b_samples])
        true_labels = np.vstack([a_labels, b_labels])

        pred_labels = opt_d(samples)

        fpr, tpr, _ = skm.roc_curve(true_labels, pred_labels)
        roc_auc = skm.auc(fpr, tpr)

        return fpr, tpr, roc_auc

    seed = 1234
    rng_key = jax.random.PRNGKey(seed)

    num_data = 1000
    data_dim = 5

    for offset in np.linspace(0, 0.5, 3):

        mean = np.zeros(data_dim)
        cov = np.eye(data_dim)
        a_samples = jax.random.multivariate_normal(
            rng_key, mean=mean, cov=cov, shape=(num_data,)
        )
        b_samples = jax.random.multivariate_normal(
            rng_key, mean=mean + offset, cov=cov, shape=(num_data,)
        )

        # Trained discriminator
        fpr, tpr, roc_auc = ROC_AUC(rng_key, a_samples, b_samples, train_nsteps=1000)
        opt_fpr, opt_tpr, opt_roc_auc = ROC_of_true_discriminator(
            rng_key, a_samples, b_samples
        )

        if offset == 0:
            label_classifier = 'classifier'
            label_optimal = 'optimal'
        else:
            label_classifier = None
            label_optimal = None
            
        plt.scatter(offset, roc_auc, color="blue", label=label_classifier)
        plt.scatter(offset, opt_roc_auc, color="orange", label=label_optimal)
        
    plt.legend()
    plt.yscale("log")
    plt.show()

    plt.plot(fpr, tpr, label="Classifier (area = %0.2f)" % roc_auc)
    plt.plot(
        np.linspace(0, 1, 10), np.linspace(0, 1, 10), linestyle="--", color="black"
    )

    # Optimal discriminator
    plt.plot(opt_fpr, opt_tpr, label="Optimal (area = %0.2f)" % opt_roc_auc)
    plt.plot(
        np.linspace(0, 1, 10), np.linspace(0, 1, 10), linestyle="--", color="black"
    )

    plt.legend(loc="lower right")
    plt.show()

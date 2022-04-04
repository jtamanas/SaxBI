import torch
import jax
import jax.numpy as np


def FlowCollate_fn(data: tuple) -> torch.Tensor:
    return np.hstack(data)


def LikelihoodRatioCollate_fn(data: tuple) -> torch.Tensor:
    """
    data comes in shuffled, so we need not re-specify the derangement
    every time and instead use something simple


    data: tuple of tuples length batch_size
            each tuple is of (x, theta) where obs and theta are tensors
            of shape (x_dim) and (theta_dim) respectively

    return: tuple with (data, label) where:
                data: is a tensor of shape (batch_size, obs_dim+theta_dim)
                label: is a tensor of shape (batch_size, 1)  (maybe just batch_size?))

    """

    batch_size = len(data)
    # the reversed index is used to reorder the shuffled data, but
    # if the batch size is odd, the middle element will be reordered.
    # we can simply avoid this by switching the middle and first elements
    reversed_idx = torch.arange(batch_size - 1, -1, step=-1)
    reversed_idx[0], reversed_idx[batch_size // 2] = (
        reversed_idx[batch_size // 2],
        reversed_idx[0],
    )
    
    x, theta = zip(*data)
    x = torch.stack(x)
    theta = torch.stack(theta)

    combo_x = torch.cat([x, x])
    combo_theta = torch.cat([theta, theta[reversed_idx]])
    labels = torch.cat([torch.ones(batch_size, 1), torch.zeros(batch_size, 1)])

    return combo_x, combo_theta, labels
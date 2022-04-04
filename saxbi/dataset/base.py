import jax
import jax.numpy as np
import numpy as onp
import torch


class BaseDataset(torch.utils.data.Dataset):
    """
    Base dataset class for all datasets.
    """

    def __init__(self, X, Theta, scale_X=None, scale_Theta=None, **kwargs):
        super(BaseDataset, self).__init__()
        assert (
            X.shape[0] == Theta.shape[0]
        ), "X and Theta must have the same number of entries"
        self.scale_X = scale_X
        self.scale_Theta = scale_Theta
        self.X = self.scale_X(X) if self.scale_X is not None else X
        self.Theta = self.scale_Theta(Theta) if self.scale_Theta is not None else Theta

    def __getitem__(self, index):
        x = self.X[index]
        theta = self.Theta[index]
        return x, theta

    def __len__(self):
        return self.X.shape[0]

import torch
import numpy as np
from torch import nn


class Softmax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.softmax = nn.Softmax(dim)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x.masked_fill(~mask, -np.inf)
        return self.softmax(x)


class SigSoftmax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.epsilon = 1e-12
        self.sigmoid = nn.LogSigmoid()
        self.softmax = nn.Softmax(dim)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x.masked_fill(~mask, -np.inf)
        return self.softmax(x + torch.log(torch.sigmoid(x) + self.epsilon))


class SphericalSoftmax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, mask=None):
        if mask is not None:
            x = x.masked_fill(~mask, 0)
        return (x ** 2) / torch.sum(x ** 2, dim=self.dim, keepdim=True)


def to_activation(name, dim):
    if name == 'softmax':
        return Softmax(dim)
    elif name == 'spherical':
        return SphericalSoftmax(dim)
    elif name == 'sigsoftmax':
        return SigSoftmax(dim)
    else:
        raise ValueError(name)

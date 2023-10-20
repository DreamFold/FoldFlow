"""Copyright (c) Dreamfold."""
import matplotlib

matplotlib.use("Agg")
import torch
import argparse

from torch import nn
from torch.utils.data import DataLoader

#import plotting
import os
import json
import ipdb

MY_EPS = {torch.float32: 1e-4, torch.float64: 1e-8}

# noinspection PyShadowingNames
def proj2manifold(x):
    u, _, vT = torch.linalg.svd(x)
    return u @ vT


# noinspection PyShadowingNames
def proj2tangent(x, v):
    shape = v.shape
    m = x.size(1)
    if v.ndim == 2:
        v = v.view(-1, m, m)
    return 0.5 * (v - x @ v.permute(0, 2, 1) @ x).view(shape)

EPS = 1e-9
MIN_NORM = 1e-15


# noinspection PyShadowingNames,PyAbstractClass
class Manifold:
    def __init__(self, ambient_dim, manifold_dim):
        """
        ambient_dim: dimension of ambient space
        manifold_dim: dimension of manifold
        """
        self.ambient_dim = ambient_dim
        self.manifold_dim = manifold_dim

    @staticmethod
    def phi(x):
        """
        x: point on ambient space
        return: point on euclidean patch
        """
        raise NotImplementedError

    @staticmethod
    def invphi(x_tilde):
        """
        x_tilde: point on euclidean patch
        return: point on ambient space
        """
        raise NotImplementedError

    @staticmethod
    def project(x):
        """
        x: manifold point on ambient space
        return: projection of x onto the manifold in the ambient space
        """
        raise NotImplementedError

    @staticmethod
    def g(x):
        """
        x: manifold point on ambient space
        return: differentiable determinant of the metric tensor at point x
        """
        raise NotImplementedError

    def norm(self, x, u, squared=False, keepdim=False):
        norm_sq = self.inner(x, u, u, keepdim)
        norm_sq.data.clamp_(MY_EPS[u.dtype])
        return norm_sq if squared else norm_sq.sqrt()
      
# noinspection PyShadowingNames,PyAbstractClass
class OrthogonalGroup(Manifold):
    def __init__(self):
        super(OrthogonalGroup, self).__init__(ambient_dim=9, manifold_dim=3)

    @staticmethod
    def proj2manifold(x):
        return proj2manifold(x)

    @staticmethod
    def proj2tangent(x, v):
        return proj2tangent(x, v)

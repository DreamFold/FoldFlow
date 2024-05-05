"""Copyright (c) Dreamfold."""
import torch
from foldflow.utils.so3_helpers import tangent_space_proj
from einops import rearrange

torch.set_default_dtype(torch.float64)


class PMLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, input):
        v = self.net(input)
        x = rearrange(input[:, :-1], "b (c d) -> b c d", c=3, d=3)
        v = rearrange(v, "b (c d) -> b c d", c=3, d=3)
        Pv = tangent_space_proj(x, v)  # Pv is on the tangent space of x
        Pv = rearrange(Pv, "b c d -> b (c d)", c=3, d=3)
        return Pv


class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=256, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GradModel(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def forward(self, x):
        x = x.requires_grad_(True)
        grad = torch.autograd.grad(torch.sum(self.action(x)), x, create_graph=True)[0]
        return grad[:, :-1]


# MLP with tangential projection of the output to the tangent space of the input
class PMLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, input):
        v = self.net(input)
        x = rearrange(input[:, :-1], "b (c d) -> b c d", c=3, d=3)
        v = rearrange(v, "b (c d) -> b c d", c=3, d=3)
        Pv = self.tangent_space_proj(x, v)  # Pvt is on the tangent space of xt
        return rearrange(Pv, "b c d -> b (c d)", c=3, d=3)

    def tangent_space_proj(self, R, M):
        """
        Project the given 3x3 matrix M onto the tangent space of SO(3) at point R in PyTorch.

        Args:
        - M (torch.Tensor): 3x3 matrix from R^9
        - R (torch.Tensor): 3x3 matrix from SO(3) representing the point of tangency

        Returns:
        - T (torch.Tensor): projected 3x3 matrix in the tangent space of SO(3) at R
        """
        # Compute the skew-symmetric part of M
        skew_symmetric_part = 0.5 * (M - M.permute(0, 2, 1))

        # Project onto the tangent space at R
        T = R @ skew_symmetric_part

        return T

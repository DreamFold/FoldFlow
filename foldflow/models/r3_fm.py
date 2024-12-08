"""
The structure of this file is greatly influenced by SE3 Diffusion by Yim et. al 2023
Link: https://github.com/jasonkyuyim/se3_diffusion
"""

from typing import Union

import numpy as np
import torch
from einops import rearrange

from foldflow.utils.condflowmatcher import ConditionalFlowMatcher


class R3FM:
    """Flow matcher for translations in R3.
    Args:
        r3_conf: R3 configuration.
        stochastic_paths: whether to use stochastic paths.
    """

    def __init__(self, r3_conf, stochastic_paths):

        self._r3_conf = r3_conf
        self.min_b = r3_conf.min_b
        self.max_b = r3_conf.max_b
        self.r3_cfm = ConditionalFlowMatcher()
        self.stochastic_paths = stochastic_paths
        self.g = r3_conf.g
        self.min_sigma = r3_conf.min_sigma

    def _scale(self, x):
        return x * self._r3_conf.coordinate_scaling

    def _unscale(self, x):
        return x / self._r3_conf.coordinate_scaling

    def b_t(self, t):
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(f"Invalid t={t}")
        return self.min_b + t * (self.max_b - self.min_b)

    def drift_coef(self, x, t):
        """Time-dependent drift coefficient."""
        return -1 / 2 * self.b_t(t) * x

    def sample_ref(self, n_samples: float = 1):
        return np.random.normal(size=(n_samples, 3))

    def marginal_b_t(self, t):
        return t * self.min_b + (1 / 2) * (t**2) * (self.max_b - self.min_b)

    def calc_trans_0(self, score_t, x_t, t, use_torch=True):
        beta_t = self.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        exp_fn = torch.exp if use_torch else np.exp
        cond_var = 1 - exp_fn(-beta_t)
        return (score_t * cond_var + x_t) / exp_fn(-1 / 2 * beta_t)

    def forward(self, x_t_1: np.ndarray, t: float, num_t: int):
        """Samples marginal p(x(t) | x(t-1)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        if not np.isscalar(t):
            raise ValueError(f"{t} must be a scalar.")
        x_t_1 = self._scale(x_t_1)
        b_t = torch.tensor(self.marginal_b_t(t) / num_t).to(x_t_1.device)
        z_t_1 = torch.tensor(np.random.normal(size=x_t_1.shape)).to(x_t_1.device)
        x_t = torch.sqrt(1 - b_t) * x_t_1 + torch.sqrt(b_t) * z_t_1
        return x_t

    def compute_sigma_t(self, t):
        if isinstance(t, float):
            t = torch.tensor(t)
        return torch.sqrt(self.g**2 * t * (1 - t) + self.min_sigma**2)

    def forward_marginal(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        x_1: Union[torch.Tensor, None] = None,
        flow_mask=None,
    ):
        """Samples marginal p(x(t) | x(0)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].
            x_1: [..., n, 3] noise translation.

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        x_0 = torch.from_numpy(x_0)
        if x_0.dim() == 3:
            seq_len = x_0.shape[1]
            x_0 = rearrange(x_0, "t s d -> (t s) d", d=3)
            t = t.repeat_interleave(seq_len)

        x_0 = self._scale(x_0)
        x_1 = torch.randn_like(x_0) if x_1 is None else torch.from_numpy(x_1)

        x_t = self.r3_cfm.sample_xt(x_0, x_1, t, epsilon=0)
        if self.stochastic_paths:
            x_t = x_t + torch.randn_like(x_t) * self.compute_sigma_t(t)

        # This seems like it should be right but its not
        x_t = x_t - x_t.mean(-2, keepdim=True)
        ut = self.r3_cfm.compute_conditional_flow(x_0, x_1, t, x_t)
        x_t = self._unscale(x_t)

        if flow_mask is not None:
            x_t = self._apply_mask(x_t, x_0, flow_mask[..., None])

            ut = self._apply_mask(
                ut, torch.zeros_like(ut).to(x_t.device), flow_mask[..., None]
            )
        return x_t, ut

    def reverse(
        self,
        *,
        x_t: np.ndarray,
        v_t: np.ndarray,
        t: float,
        dt: float,
        mask: np.ndarray = None,
        center: bool = True,
        noise_scale: float = 1.0,
    ):
        """Simulates the reverse ODE for 1 step

        Args:
            x_t: [..., 3] current positions at time t in angstroms.
            vt: [..., 3] translation vectorfield at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: True indicates which residues to update.

        Returns:
            [..., 3] positions at next step t-1.
        """
        if not np.isscalar(t):
            raise ValueError(f"{t} must be a scalar.")
        x_t = self._scale(x_t)
        perturb = -v_t * dt

        if self.stochastic_paths:
            z = noise_scale * np.random.normal(size=v_t.shape)
            perturb += self.g * np.sqrt(dt) * z

        if mask is not None:
            perturb *= mask[..., None]
        else:
            mask = np.ones(x_t.shape[:-1])
        x_t_1 = x_t + perturb
        if center:
            com = np.sum(x_t_1, axis=-2) / np.sum(mask, axis=-1)[..., None]
            x_t_1 -= com[..., None, :]
        x_t_1 = self._unscale(x_t_1)
        return x_t_1

    def vectorfield_scaling(self, t: float):
        return 1

    def vectorfield(self, x_0, x_t, t, use_torch=False, scale=False):
        if scale:
            x_t = self._scale(x_t)
            x_0 = self._scale(x_0)
        return (x_t - x_0) / (t + 1e-10)

"""
The structure of this file is greatly influenced by SE3 Diffusion by Yim et. al 2023
Link: https://github.com/jasonkyuyim/se3_diffusion
"""

import logging
from typing import Union

import numpy as np
import torch

from foldflow.utils.rigid_helpers import (
    assemble_rigid_mat,
    extract_trans_rots_mat,
)
from openfold.utils import rigid_utils as ru

from .r3_fm import R3FM
from .so3_fm import SO3FM


class SE3FlowMatcher:
    def __init__(self, se3_conf):
        self._log = logging.getLogger(__name__)
        self._se3_conf = se3_conf
        self._do_fm_rot = se3_conf.flow_rot
        self._so3_fm = SO3FM(self._se3_conf.so3, se3_conf.stochastic_paths)
        self._flow_trans = se3_conf.flow_trans
        self._r3_fm = R3FM(self._se3_conf.r3, se3_conf.stochastic_paths)
        if se3_conf.stochastic_paths:
            self._log.info("Using stochastic paths.")

        if se3_conf.ot_plan:
            self._log.info(f"Using OT plan with {self._se3_conf.ot_fn} computation.")

    def forward_marginal(
        self,
        rigids_0: ru.Rigid,
        t: float,
        flow_mask: np.ndarray = None,
        as_tensor_7: bool = True,
        rigids_1: Union[ru.Rigid, None] = None,
    ):
        """
        Args:
            rigids_0: [..., N] openfold Rigid objects
            t: continuous time in [0, 1].
            flow_mask: [..., N] which residues to flow.
            as_tensor_7:
            rigids_1: [..., N] openfold Rigid objects at time t=1 (noise).

        Returns:
            rigids_t: [..., N] noised rigid. [..., N, 7] if as_tensor_7 is true.
            trans_vectorfield: [..., N, 3] translation vectorfield
            rot_vectorfield: [..., N, 3] rotation vectorfield
            trans_vectorfield_norm: [...] translation vectorfield norm
            rot_vectorfield_norm: [...] rotation vectorfield norm
        """
        trans_0, rot_0 = extract_trans_rots_mat(rigids_0)
        rot_0 = rigids_0.get_rots().get_rot_mats().cpu().numpy()
        trans_0 = rigids_0.get_trans().cpu().numpy()

        if rigids_1 is not None:
            rot_1 = rigids_1.get_rots().get_rot_mats().cpu().numpy()
            trans_1 = rigids_1.get_trans().cpu().numpy()
        else:
            rot_1 = None
            trans_1 = None

        if not self._do_fm_rot:
            rot_t, rot_vectorfield, rot_vectorfield_scaling = (
                rot_0,
                np.zeros_like(rot_0),
                np.ones_like(t),
            )
        else:
            rot_t, rot_vectorfield = self._so3_fm.forward_marginal(
                rot_0, t, rot_1=rot_1
            )
            rot_vectorfield_scaling = self._so3_fm.vectorfield_scaling(t)

        if not self._flow_trans:
            trans_t, trans_vectorfield, trans_vectorfield_scaling = (
                trans_0,
                np.zeros_like(trans_0),
                np.ones_like(t),
            )
        else:
            trans_t, trans_vectorfield = self._r3_fm.forward_marginal(
                trans_0, t, x_1=trans_1
            )
            trans_vectorfield_scaling = self._r3_fm.vectorfield_scaling(t)

        if flow_mask is not None:
            rot_t = self._apply_mask(rot_t, rot_0, flow_mask[..., None])
            trans_t = self._apply_mask(trans_t, trans_0, flow_mask[..., None])

            trans_vectorfield = self._apply_mask(
                trans_vectorfield,
                np.zeros_like(trans_vectorfield),
                flow_mask[..., None],
            )
            rot_vectorfield = self._apply_mask(
                rot_vectorfield, np.zeros_like(rot_vectorfield), flow_mask[..., None]
            )
        rigids_t = ru.Rigid(
            rots=ru.Rotation(rot_mats=rot_t),
            trans=trans_t,
        )
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()

        return {
            "rigids_t": rigids_t,
            "trans_vectorfield": trans_vectorfield,
            "rot_vectorfield": rot_vectorfield,
            "rot_t": rot_t,
            "trans_vectorfield_scaling": trans_vectorfield_scaling,
            "rot_vectorfield_scaling": rot_vectorfield_scaling,
        }

    def calc_trans_vectorfield(self, trans_t, trans_0, t, use_torch=False, scale=True):
        return self._r3_fm.vectorfield(
            trans_t, trans_0, t, use_torch=use_torch, scale=scale
        )

    def calc_rot_vectorfield(self, rot_0, rot_t, t):
        return self._so3_fm.vectorfield(rot_0, rot_t, t)

    def _apply_mask(self, x_diff, x_fixed, diff_mask):
        return diff_mask * x_diff + (1 - diff_mask) * x_fixed

    def vectorfield_scaling(self, t):
        rot_vectorfield_scaling = self._so3_fm.vectorfield_scaling(t)
        trans_vectorfield_scaling = self._r3_fm.vectorfield_scaling(t)
        return rot_vectorfield_scaling, trans_vectorfield_scaling

    def reverse(
        self,
        rigid_t: ru.Rigid,
        rot_vectorfield: np.ndarray,
        trans_vectorfield: np.ndarray,
        t: float,
        dt: float,
        flow_mask: np.ndarray = None,
        center: bool = True,
        noise_scale: float = 1.0,
        context: torch.Tensor = None,
    ):
        """Reverse sampling function from (t) to (t-1).

        Args:
            rigid_t: [..., N] protein rigid objects at time t.
            rot_vectorfield: [..., N, 3] rotation vectorfield.
            trans_vectorfield: [..., N, 3] translation vectorfield.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: [..., N] which residues to update.
            center: true to set center of mass to zero after step

        Returns:
            rigid_t_1: [..., N] protein rigid objects at time t-1.
        """
        trans_t, rot_t = extract_trans_rots_mat(rigid_t)
        if not self._do_fm_rot:
            rot_t_1 = rot_t
        else:
            rot_t_1 = self._so3_fm.reverse(
                rot_t=rot_t,
                v_t=rot_vectorfield,
                t=t,
                dt=dt,
                noise_scale=noise_scale,
            )
        if not self._flow_trans:
            trans_t_1 = trans_t
        else:
            trans_t_1 = self._r3_fm.reverse(
                x_t=trans_t,
                v_t=trans_vectorfield,
                t=t,
                dt=dt,
                center=center,
                noise_scale=noise_scale,
            )

        if flow_mask is not None:
            trans_t_1 = self._apply_mask(trans_t_1, trans_t, flow_mask[..., None])
            rot_t_1 = self._apply_mask(rot_t_1, rot_t, flow_mask[..., None, None])
        return (rot_t_1, trans_t_1, assemble_rigid_mat(rot_t_1, trans_t_1))

    def sample_ref(
        self,
        n_samples: int,
        impute: ru.Rigid = None,
        flow_mask: np.ndarray = None,
        as_tensor_7: bool = False,
    ):
        """Samples rigids from reference distribution.

        Args:
            n_samples: Number of samples.
            impute: Rigid objects to use as imputation values if either
                translations or rotations are not flowed.
        """
        if impute is not None:
            assert impute.shape[0] == n_samples
            trans_impute, rot_impute = extract_trans_rots_mat(impute)
            trans_impute = trans_impute.reshape((n_samples, 3))
            rot_impute = rot_impute.reshape((n_samples, 3, 3))
            trans_impute = self._r3_fm._scale(trans_impute)

        if flow_mask is not None and impute is None:
            raise ValueError("Must provide imputation values.")

        if (not self._do_fm_rot) and impute is None:
            raise ValueError("Must provide imputation values.")

        if (not self._flow_trans) and impute is None:
            raise ValueError("Must provide imputation values.")

        if self._do_fm_rot:
            rot_ref = self._so3_fm.sample_ref(n_samples=n_samples)
        else:
            rot_ref = rot_impute

        if self._flow_trans:
            trans_ref = self._r3_fm.sample_ref(n_samples=n_samples)
        else:
            trans_ref = trans_impute

        if flow_mask is not None:
            rot_ref = self._apply_mask(rot_ref, rot_impute, flow_mask[..., None])
            trans_ref = self._apply_mask(trans_ref, trans_impute, flow_mask[..., None])
        trans_ref = self._r3_fm._unscale(trans_ref)
        rigids_t = assemble_rigid_mat(rot_ref, trans_ref)
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()
        return {"rigids_t": rigids_t}

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from openfold.utils import rigid_utils as ru


def extract_trans_rots(rigid: ru.Rigid):
    rot = rigid.get_rots().get_rot_mats().cpu().numpy()
    rot_shape = rot.shape
    num_rots = np.cumprod(rot_shape[:-2])[-1]
    rot = rot.reshape((num_rots, 3, 3))
    rot = Rotation.from_matrix(rot).as_rotvec().reshape(rot_shape[:-2] + (3,))
    tran = rigid.get_trans().cpu().numpy()
    return tran, rot


def extract_trans_rots_mat(rigid: ru.Rigid):
    rot = rigid.get_rots().get_rot_mats().cpu().numpy()
    tran = rigid.get_trans().cpu().numpy()
    return tran, rot


def assemble_rigid_mat(rotmat, trans):
    return ru.Rigid(
        rots=ru.Rotation(rot_mats=torch.tensor(rotmat)),
        trans=torch.tensor(trans),
    )

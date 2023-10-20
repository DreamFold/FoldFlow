"""Copyright (c) Dreamfold."""
import torch
# TODO (alex) Really scary....
# torch.set_default_dtype(torch.float64)
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import torch.nn.functional as F
from typing import Tuple
import pdb
import math

# Orthonormal basis of SO(3) with shape [3, 3, 3]
basis = torch.tensor([
    [[0.,0.,0.],[0.,0.,-1.],[0.,1.,0.]],
    [[0.,0.,1.],[0.,0.,0.],[-1.,0.,0.]],
    [[0.,-1.,0.],[1.,0.,0.],[0.,0.,0.]]])

# hat map from vector space R^3 to Lie algebra so(3)
def my_hat(v): return torch.einsum('...i,ijk->...jk', v, basis.to(v))

# Logarithmic map from SO(3) to R^3 (i.e. rotation vector)
#def Log(R): return torch.tensor(Rotation.from_matrix(R.numpy()).as_rotvec())

def Log(R): return matrix_to_axis_angle(R)
    
# logarithmic map from SO(3) to so(3), this is the matrix logarithm
def log(R): return my_hat(Log(R))

# Exponential map from so(3) to SO(3), this is the matrix exponential
def exp(A): return torch.linalg.matrix_exp(A)

# Exponential map from tangent space at R0 to SO(3)
def expmap(R0, tangent):
    skew_sym = pt_to_identity(R0, tangent)
    return R0 @ exp(skew_sym)

# Return angle of rotation. SO(3) to R^+
def Omega(R): return torch.arccos((torch.diagonal(R, dim1=-2, dim2=-1).sum(axis=-1)-1)/2)

# Power series expansion in the IGSO3 density.
def f_igso3(omega, t, L=500):
    ls = torch.arange(L)[None]  # of shape [1, L]
    return ((2*ls + 1) * torch.exp(-ls*(ls+1)*t/2) *
             torch.sin(omega[:, None]*(ls+1/2)) / torch.sin(omega[:, None]/2)).sum(dim=-1)

# IGSO3(Rt; I_3, t), density with respect to the volume form on SO(3) 
def igso3_density(Rt, t, L=500): return f_igso3(Omega(Rt), t, L)

# Marginal density of rotation angle for uniform density on SO(3)
def angle_density_unif(omega):
    return (1-torch.cos(omega))/np.pi

# Normal sample in tangent space at R0
def tangent_gaussian(R0): return torch.einsum('...ij,...jk->...ik', R0, hat(torch.randn(R0.shape[0], 3)))

# Simluation procedure for forward and reverse
def geodesic_random_walk(p_initial, drift, ts):
    Rts = {ts[0]:p_initial()}
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i-1] # negative for reverse process
        Rts[ts[i]] = expmap(Rts[ts[i-1]],
            drift(Rts[ts[i-1]], ts[i-1]) * dt + tangent_gaussian(Rts[ts[i-1]]) * np.sqrt(abs(dt)))
    return Rts

# Geodesic Distance between Rotation Matrices A and B
def geodesic_distance(A, B):
    intermed = torch.einsum('bik,bkj->bij', [torch.transpose(A, 1, 2).double(), B.double()])
    pre_distance = log(intermed)
    distance = (torch.linalg.matrix_norm(pre_distance, ord='fro') / 
                torch.sqrt(torch.tensor(2)).to(pre_distance.device))
    return distance

# Parallel Transport a matrix at v at point R to the Tangent Space at identity
def pt_to_identity(R, v):
     return (torch.transpose(R, dim0=-2, dim1=-1) @ v)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def _normalize_quaternion(quat):
  return quat / torch.norm(quat, dim=-1, keepdim=True)

def matrix_to_axis_angle(matrix):
    # Check if matrix has 3 dimensions and last two dimensions have shape 3
    if len(matrix.shape) != 3 or matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

def matrix_to_quaternion(matrix):
    num_rots = matrix.shape[0]
    matrix_diag = torch.diagonal(matrix, dim1=-2, dim2=-1)
    matrix_trace = torch.sum(matrix_diag, dim=-1, keepdim=True)
    decision = torch.cat((matrix_diag, matrix_trace), dim=-1)
    choice = torch.argmax(decision, dim=-1)
    quat = torch.zeros((num_rots, 4), dtype=matrix.dtype, device=matrix.device)

    # Indices where choice is not 3
    not_three_mask = choice != 3
    i = choice[not_three_mask]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[not_three_mask, i] = (1 - decision[not_three_mask, 3] + 2 * matrix[not_three_mask, i, i])
    quat[not_three_mask, j] = (matrix[not_three_mask, j, i] + matrix[not_three_mask, i, j])
    quat[not_three_mask, k] = (matrix[not_three_mask, k, i] + matrix[not_three_mask, i, k])
    quat[not_three_mask, 3] = (matrix[not_three_mask, k, j] - matrix[not_three_mask, j, k])

    # Indices where choice is 3
    three_mask = ~not_three_mask
    quat[three_mask, 0] = (matrix[three_mask, 2, 1] - matrix[three_mask, 1, 2])
    quat[three_mask, 1] = (matrix[three_mask, 0, 2] - matrix[three_mask, 2, 0])
    quat[three_mask, 2] = (matrix[three_mask, 1, 0] - matrix[three_mask, 0, 1])
    quat[three_mask, 3] = (1 + decision[three_mask, 3])

    return _normalize_quaternion(quat)


def quaternion_to_axis_angle(quat, degrees=False, eps=1e-6):
    quat = torch.where(quat[..., 3:4] < 0, -quat, quat)
    angle = 2. * torch.atan2(torch.norm(quat[..., :3], dim=-1), quat[..., 3])
    angle2 = angle * angle
    small_scale = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
    large_scale = angle / torch.sin(angle / 2  + eps)
    scale = torch.where(angle <= 1e-3, small_scale, large_scale)
    
    if degrees:
        scale = torch.rad2deg(scale)
    
    return scale[..., None] * quat[..., :3]

# def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
#     """
#     Convert rotations given as rotation matrices to quaternions.

#     Args:
#         matrix: Rotation matrices as tensor of shape (..., 3, 3).

#     Returns:
#         quaternions with real part first, as tensor of shape (..., 4).
#     """
#     if matrix.size(-1) != 3 or matrix.size(-2) != 3:
#         raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

#     batch_dim = matrix.shape[:-2]
#     m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
#         matrix.reshape(batch_dim + (9,)), dim=-1
#     )

#     q_abs = _sqrt_positive_part(
#         torch.stack(
#             [
#                 1.0 + m00 + m11 + m22,
#                 1.0 + m00 - m11 - m22,
#                 1.0 - m00 + m11 - m22,
#                 1.0 - m00 - m11 + m22,
#             ],
#             dim=-1,
#         )
#     )

#     # we produce the desired quaternion multiplied by each of r, i, j, k
#     quat_by_rijk = torch.stack(
#         [
#             # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
#             #  `int`.
#             torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
#             # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
#             #  `int`.
#             torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
#             # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
#             #  `int`.
#             torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
#             # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
#             #  `int`.
#             torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
#         ],
#         dim=-2,
#     )

#     # We floor here at 0.1 but the exact level is not important; if q_abs is small,
#     # the candidate won't be picked.
#     flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
#     quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

#     # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
#     # forall i; we pick the best-conditioned one (with the largest denominator)

#     return quat_candidates[
#         F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
#     ].reshape(batch_dim + (4,))


# def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
#     """
#     Convert rotations given as rotation matrices to axis/angle.

#     Args:
#         matrix: Rotation matrices as tensor of shape (..., 3, 3).

#     Returns:
#         Rotations given as a vector in axis angle form, as a tensor
#             of shape (..., 3), where the magnitude is the angle
#             turned anticlockwise in radians around the vector's
#             direction.
#     """
#     return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

# def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
#     """
#     Convert rotations given as quaternions to axis/angle.

#     Args:
#         quaternions: quaternions with real part first,
#             as tensor of shape (..., 4).

#     Returns:
#         Rotations given as a vector in axis angle form, as a tensor
#             of shape (..., 3), where the magnitude is the angle
#             turned anticlockwise in radians around the vector's
#             direction.
#     """
#     norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
#     half_angles = torch.atan2(norms, quaternions[..., :1])
#     angles = 2 * half_angles
#     eps = 1e-6
#     small_angles = angles.abs() < eps
#     sin_half_angles_over_angles = torch.empty_like(angles)
#     sin_half_angles_over_angles[~small_angles] = (
#         torch.sin(half_angles[~small_angles]) / angles[~small_angles]
#     )
#     # for x small, sin(x/2) is about x/2 - (x/2)^3/6
#     # so sin(x/2)/x is about 1/2 - (x*x)/48
#     sin_half_angles_over_angles[small_angles] = (
#         0.5 - (angles[small_angles] * angles[small_angles]) / 48
#     )
#     return quaternions[..., 1:] / sin_half_angles_over_angles

DEFAULT_ACOS_BOUND: float = 1.0 - 1e-4


def acos_linear_extrapolation(
    x: torch.Tensor,
    bounds: Tuple[float, float] = (-DEFAULT_ACOS_BOUND, DEFAULT_ACOS_BOUND),
) -> torch.Tensor:
    """
    Implements `arccos(x)` which is linearly extrapolated outside `x`'s original
    domain of `(-1, 1)`. This allows for stable backpropagation in case `x`
    is not guaranteed to be strictly within `(-1, 1)`.

    More specifically::

        bounds=(lower_bound, upper_bound)
        if lower_bound <= x <= upper_bound:
            acos_linear_extrapolation(x) = acos(x)
        elif x <= lower_bound: # 1st order Taylor approximation
            acos_linear_extrapolation(x)
                = acos(lower_bound) + dacos/dx(lower_bound) * (x - lower_bound)
        else:  # x >= upper_bound
            acos_linear_extrapolation(x)
                = acos(upper_bound) + dacos/dx(upper_bound) * (x - upper_bound)

    Args:
        x: Input `Tensor`.
        bounds: A float 2-tuple defining the region for the
            linear extrapolation of `acos`.
            The first/second element of `bound`
            describes the lower/upper bound that defines the lower/upper
            extrapolation region, i.e. the region where
            `x <= bound[0]`/`bound[1] <= x`.
            Note that all elements of `bound` have to be within (-1, 1).
    Returns:
        acos_linear_extrapolation: `Tensor` containing the extrapolated `arccos(x)`.
    """

    lower_bound, upper_bound = bounds

    if lower_bound > upper_bound:
        raise ValueError("lower bound has to be smaller or equal to upper bound.")

    if lower_bound <= -1.0 or upper_bound >= 1.0:
        raise ValueError("Both lower bound and upper bound have to be within (-1, 1).")

    # init an empty tensor and define the domain sets
    acos_extrap = torch.empty_like(x)
    x_upper = x >= upper_bound
    x_lower = x <= lower_bound
    x_mid = (~x_upper) & (~x_lower)

    # acos calculation for upper_bound < x < lower_bound
    acos_extrap[x_mid] = torch.acos(x[x_mid])
    # the linear extrapolation for x >= upper_bound
    acos_extrap[x_upper] = _acos_linear_approximation(x[x_upper], upper_bound)
    # the linear extrapolation for x <= lower_bound
    acos_extrap[x_lower] = _acos_linear_approximation(x[x_lower], lower_bound)

    return acos_extrap

def _acos_linear_approximation(x: torch.Tensor, x0: float) -> torch.Tensor:
    """
    Calculates the 1st order Taylor expansion of `arccos(x)` around `x0`.
    """
    return (x - x0) * _dacos_dx(x0) + math.acos(x0)


def _dacos_dx(x: float) -> float:
    """
    Calculates the derivative of `arccos(x)` w.r.t. `x`.
    """
    return (-1.0) / math.sqrt(1.0 - x * x)

def so3_relative_angle(
    R1: torch.Tensor,
    R2: torch.Tensor,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates the relative angle (in radians) between pairs of
    rotation matrices `R1` and `R2` with `angle = acos(0.5 * (Trace(R1 R2^T)-1))`

    .. note::
        This corresponds to a geodesic distance on the 3D manifold of rotation
        matrices.

    Args:
        R1: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        R2: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        cos_angle: If==True return cosine of the relative angle rather than
            the angle itself. This can avoid the unstable calculation of `acos`.
        cos_bound: Clamps the cosine of the relative rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.
        eps: Tolerance for the valid trace check of the relative rotation matrix
            in `so3_rotation_angle`.
    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R1` or `R2` is of incorrect shape.
        ValueError if `R1` or `R2` has an unexpected trace.
    """
    R12 = R1.double() @ R2.permute(0, 2, 1).double()
    return so3_rotation_angle(R12, cos_angle=cos_angle, cos_bound=cos_bound, eps=eps)


def so3_rotation_angle(
    R: torch.Tensor,
    eps: float = 1e-4,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates angles (in radians) of a batch of rotation matrices `R` with
    `angle = acos(0.5 * (Trace(R)-1))`. The trace of the
    input matrices is checked to be in the valid range `[-1-eps,3+eps]`.
    The `eps` argument is a small constant that allows for small errors
    caused by limited machine precision.

    Args:
        R: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: Tolerance for the valid trace check.
        cos_angle: If==True return cosine of the rotation angles rather than
            the angle itself. This can avoid the unstable
            calculation of `acos`.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.

    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        raise ValueError("A matrix has trace outside valid range [-1-eps,3+eps].")

    # phi ... rotation angle
    phi_cos = (rot_trace - 1.0) * 0.5

    if cos_angle:
        return phi_cos
    else:
        if cos_bound > 0.0:
            bound = 1.0 - cos_bound
            return acos_linear_extrapolation(phi_cos, (-bound, bound))
        else:
            return torch.acos(phi_cos)


        
def so3_exp_map(log_rot: torch.Tensor, eps: float = 0.0001) -> torch.Tensor:
    """
    Convert a batch of logarithmic representations of rotation matrices `log_rot`
    to a batch of 3x3 rotation matrices using Rodrigues formula [1].

    In the logarithmic representation, each rotation matrix is represented as
    a 3-dimensional vector (`log_rot`) who's l2-norm and direction correspond
    to the magnitude of the rotation angle and the axis of rotation respectively.

    The conversion has a singularity around `log(R) = 0`
    which is handled by clamping controlled with the `eps` argument.

    Args:
        log_rot: Batch of vectors of shape `(minibatch, 3)`.
        eps: A float constant handling the conversion singularity.

    Returns:
        Batch of rotation matrices of shape `(minibatch, 3, 3)`.

    Raises:
        ValueError if `log_rot` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    return _so3_exp_map(log_rot, eps=eps)[0]

def _so3_exp_map(
    log_rot: torch.Tensor, eps: float = 0.0001
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A helper function that computes the so3 exponential map and,
    apart from the rotation matrix, also returns intermediate variables
    that can be re-used in other functions.
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)

    R = (
        fac1[:, None, None] * skews
        # pyre-fixme[16]: `float` has no attribute `__getitem__`.
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    return R, rot_angles, skews, skews_square

def so3_log_map(
    R: torch.Tensor, eps: float = 0.0001, cos_bound: float = 1e-4
) -> torch.Tensor:
    """
    Convert a batch of 3x3 rotation matrices `R`
    to a batch of 3-dimensional matrix logarithms of rotation matrices
    The conversion has a singularity around `(R=I)` which is handled
    by clamping controlled with the `eps` and `cos_bound` arguments.

    Args:
        R: batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: A float constant handling the conversion singularity.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call when computing `so3_rotation_angle`.
            Note that the non-finite outputs/gradients are returned when
            the rotation angle is close to 0 or π.

    Returns:
        Batch of logarithms of input rotation matrices
        of shape `(minibatch, 3)`.

    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    phi = so3_rotation_angle(R, cos_bound=cos_bound, eps=eps)

    phi_sin = torch.sin(phi)

    # We want to avoid a tiny denominator of phi_factor = phi / (2.0 * phi_sin).
    # Hence, for phi_sin.abs() <= 0.5 * eps, we approximate phi_factor with
    # 2nd order Taylor expansion: phi_factor = 0.5 + (1.0 / 12) * phi**2
    phi_factor = torch.empty_like(phi)
    ok_denom = phi_sin.abs() > (0.5 * eps)
    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    phi_factor[~ok_denom] = 0.5 + (phi[~ok_denom] ** 2) * (1.0 / 12)
    phi_factor[ok_denom] = phi[ok_denom] / (2.0 * phi_sin[ok_denom])

    log_rot_hat = phi_factor[:, None, None] * (R - R.permute(0, 2, 1))

    log_rot = hat_inv(log_rot_hat)

    return log_rot



def hat_inv(h: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse Hat operator [1] of a batch of 3x3 matrices.

    Args:
        h: Batch of skew-symmetric matrices of shape `(minibatch, 3, 3)`.

    Returns:
        Batch of 3d vectors of shape `(minibatch, 3, 3)`.

    Raises:
        ValueError if `h` is of incorrect shape.
        ValueError if `h` not skew-symmetric.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim1, dim2 = h.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    ss_diff = torch.abs(h + h.permute(0, 2, 1)).max()

    HAT_INV_SKEW_SYMMETRIC_TOL = 1e-5
    if float(ss_diff) > HAT_INV_SKEW_SYMMETRIC_TOL:
        print('skewsym error', ss_diff)
        raise ValueError("One of input matrices is not skew-symmetric.")

    x = h[:, 2, 1]
    y = h[:, 0, 2]
    z = h[:, 1, 0]

    v = torch.stack((x, y, z), dim=1)

    return v


def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`

    Raises:
        ValueError if `v` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h

def tangent_space_proj(R, M):
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

def norm_SO3(R, T_R):
    # calulate the norm squared of matrix T_R in the tangent space of R
    r = pt_to_identity(R, T_R)                                  # matrix r is in so(3)
    norm = -torch.diagonal(r@r, dim1=-2, dim2=-1).sum(dim=-1)/2 #-trace(rTr)/2
    return norm

def norm_SO3_aa(R, T_R):
    # calulate the norm squared of matrix T_R in the tangent space of R
    r = pt_to_identity(R, T_R) # matrix r is in so(3)
    r_aa = hat_inv(r)          # r_aa is the axis-angle representation of r
    norm = torch.linalg.norm(r_aa, dim=-1)**2
    return norm


def check_skew_sym(h):
    # check if matrix h is skew-symmetric
    SKEW_SYMMETRIC_TOL = 1e-4
    skew_sym_eq = torch.allclose(h, -h.permute(0, 2, 1), atol=SKEW_SYMMETRIC_TOL, rtol=0)
    if not skew_sym_eq:
        print('skew symmetric error', torch.abs(h+h.permute(0, 2, 1)).max())
        return False
    return True
    
    
def check_rot_mat(R):
    # check if matrix R is a rotation matrix
    ROT_MAT_TOL = 1e-4
    rot_eq = torch.allclose(torch.inverse(R), R.permute(0, 2, 1), atol=ROT_MAT_TOL, rtol=0)
    rot_det_eq = torch.allclose(torch.det(R), torch.ones_like(torch.det(R)), atol=ROT_MAT_TOL, rtol=0)    
    if not (rot_eq and rot_det_eq):
        return False
    return True
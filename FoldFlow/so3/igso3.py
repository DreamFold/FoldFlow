"""Copyright (c) Dreamfold."""
import torch
from torch import Tensor
from functorch import vmap
from .so3_helpers import so3_exp_map

def f_igso3_small(omega, sigma):
    """ Borrowed from: https://github.com/tomato1mule/edf/blob/1dd342e849fcb34d3eb4b6ad2245819abbd6c812/edf/dist.py#L99
    This function implements the approximation of the density function of omega of the isotropic Gaussian distribution. 
    """
    # TODO: check for stability and maybe replace by limit in 0 for small values

    #TODO: figure out scaling constant: eps = eps/2
    eps = (sigma / torch.sqrt(torch.tensor([2])).to(device=omega.device))**2

    pi = torch.Tensor([torch.pi]).to(device=omega.device)

    small_number = 1e-9
    small_num = small_number / 2 
    small_dnm = (1-torch.exp(-1. * pi**2 / eps)*(2  - 4 * (pi**2) / eps)) * small_number

    return (0.5 * torch.sqrt(pi) * (eps ** -1.5) * 
            torch.exp((eps - (omega**2 / eps))/4) / (torch.sin(omega/2) + small_num) *
            (small_dnm + omega - ((omega - 2*pi)*torch.exp(pi * (omega - pi) / eps) 
                                   + (omega + 2*pi)*torch.exp(-pi * (omega+pi) / eps))))


# Marginal density of rotation angle for uniform density on SO(3)
def angle_density_unif(omega):
    return (1-torch.cos(omega))/torch.pi

def interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1]) # slope
    b = fp[:-1] - (m * xp[:-1])                 # y-intercept

    indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), dim=1) - 1 
    indicies = torch.clamp(indicies, 0, len(m) - 1)

    return m[indicies] * x + b[indicies]


def _f(omega, eps):
    return f_igso3_small(omega, eps)

def _pdf(omega, eps):
    f_unif = angle_density_unif(omega)
    return _f(omega, eps) * f_unif

def _sample(eps, n):
    # sample n points from IGSO3(I, eps)
    num_omegas = 1024
    omega_grid = torch.linspace(0, torch.pi, num_omegas+1).to(eps.device)[1:]  # skip omega=0
    # numerical integration of (1-cos(omega))/pi*f_igso3(omega, eps) over omega    
    pdf = _pdf(omega_grid, eps)
    dx = omega_grid[1] - omega_grid[0]
    cdf = torch.cumsum(pdf, dim=-1) * dx # cumalative density function

    # sample n points from the distribution
    rand_angle = torch.rand(n).to(eps.device)
    omegas = interp(rand_angle, cdf, omega_grid) 
    axes = torch.randn(n, 3).to(eps.device) #sample axis uniformly 
    axis_angle = omegas[..., None] * axes / torch.linalg.norm(axes, dim=-1, keepdim=True)
    return axis_angle

def _batch_sample(mu, eps, n):
   aa_samples = vmap(_sample, in_dims=(0, None), randomness="different")(eps, n).squeeze()
   return mu @ so3_exp_map(aa_samples)
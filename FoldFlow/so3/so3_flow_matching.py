"""Copyright (c) Dreamfold."""
import torch
from scipy.spatial.transform import Rotation
from geomstats._backend import _backend_config as _config
#_config.DEFAULT_DTYPE = torch.cuda.FloatTensor
from FoldFlow.utils.optimal_transport import SO3OTPlanSampler
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from .so3_helpers import *
from einops import rearrange
from functorch import vmap
from .igso3 import _batch_sample


# Riemannian gradient of f at R using automatic differentiation
def riemannian_gradient(f, R):
    coefficients = torch.zeros(list(R.shape[:-2])+[3], requires_grad=True).to(R.device)
    R_delta  = expmap(R, R @ hat(coefficients))
    grad_coefficients = torch.autograd.grad(f(R_delta).sum(), coefficients, )[0]
    return R @ hat(grad_coefficients)

class SO3ConditionalFlowMatcher:
    """
    Class to compute the FoldFlow-base method. It is the parent class for the 
    FoldFlow-OT and FoldFlow-SFM methods. For sake of readibility, the doc of 
    most function only describes the function purpose. The documentation 
    describing all inputs can be found in the 
    sample_location_and_conditional_flow class.
    """

    def __init__(self, manifold):
        self.sigma = None
        self.manifold = manifold
        self.vec_manifold = SpecialOrthogonal(n=3, point_type="vector")
    
    def vec_log_map(self, x0, x1, if_matrix_format=False):
        """
        Function which compute the SO(3) log map efficiently.
        """
        # get logmap of x_1 from x_0
        # convert to axis angle to compute logmap efficiently
        if if_matrix_format:
            rot_x0 = matrix_to_axis_angle(x0) 
            rot_x1 = matrix_to_axis_angle(x1)
        else:
            rot_x0 = x0
            rot_x1 = x1
        log_x1 = self.vec_manifold.log_not_from_identity(rot_x1, rot_x0)
        return log_x1, rot_x0
        
    def sample_xt(self, x0, x1, t, if_matrix_format=False):
        """
        Function which compute the sample xt along the geodesic from x0 to x1 on SO(3).
        """
        # sample along the geodesic from x0 to x1
        log_x1, rot_x0 = self.vec_log_map(x0, x1, if_matrix_format=if_matrix_format)
        # group exponential at x0
        xt = self.vec_manifold.exp_not_from_identity(t.reshape(-1, 1) * log_x1, rot_x0)
        xt = self.vec_manifold.matrix_from_rotation_vector(xt)
        return xt

    def compute_conditional_flow_simple(self, t, xt):
        """
        Function which computes the vector field through the sample xt's time derivative
        for simple manifold.
        """
        xt = rearrange(xt, 'b c d -> b (c d)', c=3, d=3)
        def index_time_der(i):
            return torch.autograd.grad(xt, t, i, create_graph=True, retain_graph=True)[0]
        xt_dot = vmap(index_time_der, in_dims=1)(torch.eye(9).to(xt.device).repeat(xt.shape[0], 1, 1))
        return rearrange(xt_dot, '(c d) b -> b c d', c=3, d=3)
    
    def compute_conditional_flow(self, xt, x0, x1, t):
        """
        Function which computes the general vector field for k(t) = 1-t.
        """
        # compute the geodesic distance
        dist_x0_x1 = geodesic_distance(x0, x1)                       # d(x0, x1)
        geo_dist = lambda x: geodesic_distance(x, x1)
        dist_grad_wrt_xt = riemannian_gradient(geo_dist, xt) # nabla_xt d(xt, x1)

        # Compute the geodesic norm ||.||_g:
        denom_term = norm_SO3(xt, dist_grad_wrt_xt)

        output = -dist_x0_x1[:, None, None] * dist_grad_wrt_xt / denom_term[:, None, None]
        return output #* 2 * t[:, None, None]
        
    def sample_location_and_conditional_flow(self, x0, x1, time_der=True):
        """
        Compute the sample xt along the geodesic from x0 to x1 (see Eq.2 [1])
        and the conditional vector field ut(xt|z). The coupling q(x0,x1) is
        the independent coupling.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        time_der : bool
            ut computed through time derivative


        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn along the geodesic
        ut : conditional vector field ut(xt|z)

        References
        ----------
        [1] SE(3)-Stochastic Flow Matching for Protein Backbone Generation, Bose et al.
        [2] Riemannian Flow Matching on General Geometries, Chen et al.
        """
        t = torch.rand(x0.shape[0]).type_as(x0).to(x0.device)
        t.requires_grad = True
        xt = self.sample_xt(x0, x1, t, if_matrix_format=True)
        if time_der:
            delta_r = torch.transpose(x0, dim0=-2, dim1=-1) @ xt.double()
            ut = xt @ log(delta_r)/t[:, None, None]
            # Above is faster than taking the time derivative like in [2]
            # ut = self.compute_conditional_flow_simple(t, xt)
        else:
            # Compute general vector field like in [2] 
            ut = self.compute_conditional_flow(xt, x0, x1, t)
        return t, xt, ut


class SO3OptimalTransportConditionalFlowMatcher(SO3ConditionalFlowMatcher):
    """Child class for optimal transport FoldFlow method. This class implements
    FoldFlow-OT method and inherits the SO3ConditionalFlowMatcher parent class.
    This idea is based on OT-CFM [3].

    It overrides the sample_location_and_conditional_flow.
    """

    def __init__(self, sigma: float = 0.0, manifold=None):
        super().__init__(manifold)
        self.sigma = sigma
        self.ot_sampler = SO3OTPlanSampler(method="exact")

    def sample_location_and_conditional_flow(self, x0, x1, time_der=True):
        """
        Compute the sample xt along the geodesic from x0 to x1 (see Eq.2 [1])
        and the conditional vector field ut(xt|z). The coupling q(x0,x1) is
        the minibatch OT coupling [3,4].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        time_der : bool
            ut computed through time derivative


        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn along the geodesic
        ut : conditional vector field ut(xt|z)

        References
        ----------
        [1] SE(3)-Stochastic Flow Matching for Protein Backbone Generation, Bose et al.
        [3] Improving and generalizing flow-based generative models with 
        minibatch optimal transport, Tong et al.
        [4] Learning with minibatch Wasserstein: asymptotic and gradient properties, Fatras et al.
        """
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0.double(), x1.double(), time_der=time_der)


class SO3SFM(SO3ConditionalFlowMatcher):
    """Child class for stochastic and optimal transport FoldFlow method. This class implements
    the FoldFlow-SFM method and inherits the SO3ConditionalFlowMatcher parent class.
    
    For sake of readibility, the doc of most function only describes the function purpose. The
    documentation  describing all inputs can be found in the 
    sample_location_and_conditional_flow class.

    It overrides the compute_conditional_flow, sample_location_and_conditional_flow functions

    """
    def __init__(self, manifold=None):
        super().__init__(manifold)
        self.epsilon = None
        self.manifold = manifold
        self.ot_sampler = SO3OTPlanSampler(method="exact")
        self.g = 0.1
    
    def compute_mu_t(self, x0, x1, t):
        """
        Function which compute the IGSO3 mean (see Eq (9) in [1]).
        """
        return super().sample_xt(x0.double(), x1.double(), t, if_matrix_format=True)
    
    def compute_epsilon_t(self, t):
        """
        Function which compute the IGSO3 standard deviation (see Eq (9) in [1]).
        """
        sigma_t = self.g * torch.sqrt(t * (1-t) + 1e-4)
        return torch.clamp(sigma_t, 0.01)
    
    def sample_zt(self, x0, x1, t):
        """
        Function which compute the sample zt along the geodesic from x0 to x1 on SO(3)
        following Eq (9) in [1].
        """
        xt = self.compute_mu_t(x0, x1, t) # geodesic interpolation (mean of igso3)
        epsilon_t = self.compute_epsilon_t(t) # g^2 t (1-t)
        return _batch_sample(xt, epsilon_t, 1).detach()
    
    def compute_conditional_flow(self, zt, x1, t):
        """
        Function which computes the vector field for a sample zt on SO3.
        """
        x1_minus_zt = torch.transpose(zt, dim0=-2, dim1=-1) @ x1.double()
        ut = zt @ log(x1_minus_zt)/(1-t[:, None, None])
        return ut

    def sample_location_and_conditional_flow(self, x0, x1):
        """
        Compute the sample xt along the geodesic from x0 to x1 (see Eq.9 [1])
        and the conditional vector field ut(xt|z).

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        time_der : bool
            ut computed through time derivative

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn along the geodesic
        ut : conditional vector field ut(xt|z)

        References
        ----------
        [1] SE(3)-Stochastic Flow Matching for Protein Backbone Generation, Bose et al.
        """
        t = torch.clamp(torch.rand(x0.shape[0]).type_as(x0).to(x0.device), min=0.01, max=0.99)
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        zt = self.sample_zt(x0, x1, t)
        ut = self.compute_conditional_flow(zt, x1, t)
        return t, zt, ut

"""Copyright (c) Dreamfold."""
import torch
from scipy.spatial.transform import Rotation
from geomstats._backend import _backend_config as _config
from .optimal_transport import SO3OTPlanSampler
from foldflow.utils.so3_helpers import so3_relative_angle, log
from foldflow.utils.so3_condflowmatcher import SO3ConditionalFlowMatcher
from einops import rearrange
from functorch import vmap
from foldflow.utils.igso3 import _batch_sample


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

    def sample_location_and_conditional_flow(self, x0, x1):
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
        x0, x1 = self.ot_sampler.sample_plan(x0.double(), x1.double())
        return super().sample_location_and_conditional_flow_simple(
            x0.double(), x1.double()
        )


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
        return super().sample_xt(x0.double(), x1.double(), t)

    def compute_epsilon_t(self, t):
        """
        Function which compute the IGSO3 standard deviation (see Eq (9) in [1]).
        """
        sigma_t = self.g * torch.sqrt(t * (1 - t) + 1e-4)
        return torch.clamp(sigma_t, 0.01)

    def sample_zt(self, x0, x1, t):
        """
        Function which compute the sample zt along the geodesic from x0 to x1 on SO(3)
        following Eq (9) in [1].
        """
        xt = self.compute_mu_t(x0, x1, t)  # geodesic interpolation (mean of igso3)
        epsilon_t = self.compute_epsilon_t(t).to(x0.device)  # g^2 t (1-t)
        return _batch_sample(xt, epsilon_t, 1).detach()

    def compute_conditional_flow(self, zt, x1, t):
        """
        Function which computes the vector field for a sample zt on SO3.
        """
        x1_minus_zt = torch.transpose(zt, dim0=-2, dim1=-1) @ x1.double()
        ut = zt @ log(x1_minus_zt) / (1 - t[:, None, None])
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
        t = torch.clamp(
            torch.rand(x0.shape[0]).type_as(x0).to(x0.device), min=0.01, max=0.99
        )
        x0, x1 = self.ot_sampler.sample_plan(x0.double(), x1.double())
        zt = self.sample_zt(x0, x1, t)
        ut = self.compute_conditional_flow(zt, x1, t)
        return t, zt, ut

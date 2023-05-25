from gpytorch.likelihoods import MultitaskGaussianLikelihood
from distributions.multitask_multivariate_normal_mask import MultitaskMultivariateNormalMask
from gpytorch.lazy import LazyEvaluatedKernelTensor
from typing import Any 

######################################################################
class MultitaskGaussianLikelihoodMask(MultitaskGaussianLikelihood):
    r"""
    A subclass of MultitaskGaussianLikelihood, instantiated with the
    same parameters (so consult that documentation for arguments). The
    only difference is that the marginal() method expects to be called
    with a MultitaskMultivariateNormalMask argument instead of with a
    MultitaskMultivariateNormal argument, and will pass the mask to the
    MultitaskMultivariateNormalMask output.
    """

#################
    def marginal(self, function_dist: MultitaskMultivariateNormalMask,
                 *params: Any, **kwargs: Any
                 ) -> MultitaskMultivariateNormalMask:
        r"""
        If :math:`\text{rank} = 0`, adds the task noises to the diagonal of the
        covariance matrix of the supplied
        :obj:`~gpytorch.distributions.MultitaskMultivariateNormalMask`.  Otherwise,
        adds a rank `rank` covariance matrix to it.
        
        """

        mean, covar = function_dist.mean, function_dist.covar_orig
        mask = function_dist.mask

        # ensure that sumKroneckerLT is actually called
        if isinstance(covar, LazyEvaluatedKernelTensor):
            covar = covar.evaluate_kernel()

        covar_kron_lt = self._shaped_noise_covar(
            mean.shape, add_noise=self.has_global_noise, interleaved=function_dist._interleaved
        )
        covar = covar + covar_kron_lt

        return function_dist.__class__(mask, mean, covar, interleaved=function_dist._interleaved)


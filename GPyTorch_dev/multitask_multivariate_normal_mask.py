import torch
from torch import Tensor, BoolTensor
from gpytorch.distributions import MultitaskMultivariateNormal
from linear_operator import LinearOperator

######################################################################
class MultitaskMultivariateNormalMask(MultitaskMultivariateNormal):
    r"""
    Constructs a multi-output multivariate Normal random variable, with a Boolean
    mask to mask out missing target values.

    Args:

    :param torch.BoolTensor mask: An n x t boolean tensor whose False entries indicate
        missing training data target values whose corresponding rows and columns should be omitted from
        the mean and covariance
    :param torch.Tensor mean:  An `n x t` or batch `b x n x t` matrix of means for the MVN distribution.
    :param ~linear_operator.operators.LinearOperator covar: An `... x (nt) x (nt)` (batch) matrix.
        covariance matrix of MVN distribution.
    :param bool validate_args: (default=False) If True, validate `mean` anad `covariance_matrix` arguments.
    :param bool interleaved: (default=True) If True, covariance matrix is interpreted as block-diagonal w.r.t.
        inter-task covariances for each observation. If False, it is interpreted as block-diagonal
        w.r.t. inter-observation covariance for each task.

    Note that while the mask operates on batches, a single mask is common to all batches. This is necessary,
    because the batching mechanism only supports batches of the same size, whereas different batch masks might
    require batches of different sizes.
    """

#################
    def __init__(self,
                 mask: BoolTensor,
                 mean: Tensor, 
                 covariance_matrix: Tensor, 
                 validate_args: bool = False, 
                 interleaved: bool = True) -> None:
        
        if not isinstance(mask, BoolTensor):
            raise ValueError("mask must be a BoolTensor")
        
        if len(mask.size) != 2:
            raise ValueError("mask must be a 2-D BoolTensor")

        if not torch.is_tensor(mean) and not isinstance(mean, LinearOperator):
            raise RuntimeError("The mean of a MultitaskMultivariateNormal must be a Tensor or LinearOperator")

        if not torch.is_tensor(covariance_matrix) and not isinstance(covariance_matrix, LinearOperator):
            raise RuntimeError("The covariance of a MultitaskMultivariateNormal must be a Tensor or LinearOperator")

        if mean.dim() < 2:
            raise RuntimeError("mean should be a matrix or a batch matrix (batch mode)")

        # Ensure that shapes are broadcasted appropriately across the mean and covariance
        # Means can have singleton dimensions for either the `n` or `t` dimensions
        batch_shape = torch.broadcast_shapes(mean.shape[:-2], covariance_matrix.shape[:-2])
        if mean.shape[-2:].numel() != covariance_matrix.size(-1):
            if covariance_matrix.size(-1) % mean.shape[-2:].numel():
                raise RuntimeError(
                    f"mean shape {mean.shape} is incompatible with covariance shape {covariance_matrix.shape}"
                )
            elif mean.size(-2) == 1:
                mean = mean.expand(*batch_shape, covariance_matrix.size(-1) // mean.size(-1), mean.size(-1))
            elif mean.size(-1) == 1:
                mean = mean.expand(*batch_shape, mean.size(-2), covariance_matrix.size(-2) // mean.size(-2))
            else:
                raise RuntimeError(
                    f"mean shape {mean.shape} is incompatible with covariance shape {covariance_matrix.shape}"
                )
        else:
            mean = mean.expand(*batch_shape, *mean.shape[-2:])

        # mean.shape[-2] and mask.shape[0] should be the same for training, but differ during prediction, when
        # mean.shape[-2] is larger than mask.shape[0]. mean.shape[-1] and mask.shape[1] should always agree.
        if  mask.shape[1] != mean.shape[-1]:
            raise RuntimeError("The value of mask.shape[1] must be equal to the number of tasks")
        elif mean.shape[-2] < mask.shape[-2]:
            raise RuntimeError("The value of mask.shape[0] should not exceed the number of training samples")

        
        self._output_shape = mean.shape
        # TODO: Instead of transpose / view operations, use a PermutationLinearOperator (see #539)
        # to handle interleaving
        self._interleaved = interleaved
        if self._interleaved:
            mean_mvn = mean.reshape(*mean.shape[:-2], -1)
            mask_mvn = mask.flatten()
        else:
            mean_mvn = mean.transpose(-1, -2).reshape(*mean.shape[:-2], -1)
            mask_mvn = mask.transpose(0, 1).flatten()

        ntrain = len(mask_mvn)
        idx = torch.arange(ntrain, dtype=torch.long)[mask_mvn]
        if mean.shape[-2] > ntrain:
            idx2 = torch.arange(ntrain, mean.shape[-1], dtype=torch.long)
            idx = torch.cat(idx, idx2)
        cols = torch.stack([idx] * len(idx), dim=0)
        rows = cols.transpose(0,1)

        mean_mvn = mean_mvn[...,idx]
        covariance_matrix = covariance_matrix[..., rows, cols]

        super(MultitaskMultivariateNormal, self).__init__(mean=mean_mvn, covariance_matrix=covariance_matrix, validate_args=validate_args)

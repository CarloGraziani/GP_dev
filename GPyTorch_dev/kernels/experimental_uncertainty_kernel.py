from gpytorch.kernels import Kernel
import torch
from torch import Tensor
from linear_operator.operators.block_diag_linear_operator import BlockDiagLinearOperator
from linear_operator.operators.zero_linear_operator import ZeroLinearOperator
from copy import deepcopy
from typing import Tuple, Union, Optional

def fix_lengthscale(parent_cls):
    """
    Ensure that a stationary kernel may be defined without it acquiring an adjustable lengthscale.
    """
    class kernel_cls(parent_cls):
        def __init__(self, *arg, **kwargs):
            super().__init__(*arg, **kwargs)

    kernel_cls.is_stationary = True
    kernel_cls.has_lengthscale = False
    kernel_cls.lengthscale = Tensor([1.0])
                                    
    return kernel_cls

SKernel = fix_lengthscale(Kernel)

class ExperimentalUncertaintyKernel(SKernel):
    """
    Kernel for characterizing uncertainty due to experiment-to-experiment
    irreproducibility.

    The idea is that training is performed using a set of independent
    experiments, some of which may be performed at replicated settings, but
    which differ in outcome from their companion experiments at the same
    settings because of uncontrolled experimental factors. This kernel is an
    additive block-diagonal term, in which each block represents the data due
    to a single experiment.

    Each experiment in the training data set is assumed to produce data
    representable by a 1-dimensional input and a 1-dimensional output (for
    example, a spectrum as a function of wavenumber). There are also :math:`c`
    control parameters, so that the dimensionality of the full input space
    (i.e. train_x.shape[-1]) is :math:`c+1`.

    In order to distinguish training and evaluation/testing diagonal blocks
    from each other, this kernel needs the train_x array at instantiation,
    so that the forward() method can tell requests for the train-train
    covariance from those for train-test covariance (all zero) and the
    test-test covariance.

    At present, test/prediction is implemented assuming that one predicts a
    single experiment---that is, the coordinates in test_x[...,:] that correspond
    to the :math:`c` control parameters are the same across all samples.

    Args:
        :param base_kernel: kernel that will furnish the within-experiment model
        :type base_kernel: Kernel

        :param train_x: The training data input array.
        :type train_x: Tensor

        :param data_size: If a single int, the size of training data from a
            single experiment (number of samples). This must be the same for
            all experiments. The number of experiments is determined from the
            input data array train_x, which must have train_x.shape[-2] equal
            to a multiple of data_size.
            
            If a tuple of ints, these are the individual training data sizes of
            the entire list of experiments. The input data must then satisfy
            train_x.shape[-2] equal to the sum of these ints.
        :type data_size: int or tuple of int

        :param exp_par_ind: The indices of the input vector that correspond to
            controlled experimental input parameters. The input data
            dimensionality train_x.shape[-1] must be equal to
            len(exp_par_ind)+1, which is to say that each experiment produces
            1-dimensional data.
        :type exp_par_int: tuple of int

        :param outputscale_fn: Whether the log output scale should be a linear
            function of the controlled experimental parameters.
        :type outputscale_fn: Bool, Optional, default=False

        :param lengthscale_fn: Whether the log lengthscale should be a linear
            function of the controlled experimental parameters.
        :type lengthscale_fn: Bool, Optional, default=False

    """

#################
    def __init__(self,
                 base_kernel: Kernel,
                 train_x: Tensor,
                 data_size: Union[int, Tuple[int]],
                 exp_par_ind: Tuple[int],
                 outputscale_fn: Optional[bool] = False,
                 lengthscale_fn: Optional[bool] = False,
                 **kwargs) -> None:

        super(ExperimentalUncertaintyKernel, self).__init__(**kwargs)

        self.is_stationary = base_kernel.is_stationary
        self.base_kernel = base_kernel
        self.train_x = train_x
        self.exp_par_ind = exp_par_ind
        epl = len(exp_par_ind)
        self.outputscale_fn = outputscale_fn
        self.lengthscale_fn = lengthscale_fn

        n_training_samples = train_x.shape[-2]
        if isinstance(data_size, int):
            if n_training_samples % data_size != 0:
                raise ValueError("The number of training data samples is {n_training_samples:%d}, which is not a multiple of the data size {data_size}.")
            self.n_exp = n_training_samples // data_size
            self.data_size = data_size
        elif isinstance(data_size, tuple):
            self.data_size = Tensor(data_size).to(int)
            n_pts = self.data_size.sum()
            if n_pts != n_training_samples:
                raise ValueError("The number of training samples in data_size ({n_pts}) does not match the number in train_x ({n_training_samples}).")
            self.n_exp = len(data_size)
        else:
            raise ValueError("data_size must be int or tuple of int.")

        self.register_parameter("os_0", 
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape,1)))

        self.register_parameter("ls_0", 
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape,1)))
        
        if outputscale_fn:
            self.register_parameter("os_v", 
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, epl)))
        else:
            self.register_buffer("os_v", 
                                tensor=torch.nn.Parameter(torch.zeros(*self.batch_shape, epl)))

        if lengthscale_fn:
            self.register_parameter("ls_v", 
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, epl)))
        else:
            self.register_buffer("ls_v", 
                                tensor=torch.nn.Parameter(torch.zeros(*self.batch_shape, epl)))

#################
    def forward(self, x1: Tensor, x2: Tensor, last_dim_is_batch: bool = False, 
                diag: bool = False, **params) -> Tensor:
        
        x1_ = x1.clone() ; x2_ = x2.clone()
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -3)
            x2_ = x2_.transpose(-1, -3)

        try:
            bs = torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2]) # Are batch shapes compatible?
        except RuntimeError:
            raise ValueError("Batch shapes of x1 and x2 are incompatible")

        idim =   len(self.exp_par_ind) + 1      
        if not x1_.shape[-1] == idim or not x2_.shape[-1] == idim:
            raise ValueError("Only 1-D within-experiment data supported") # this may change someday
        
        if not torch.equal(x1_, x2_): # Train-test covariance case
            # return torch.zeros(tuple([*bs, x1_.shape[-2], x2_.shape[-2]]))
            return ZeroLinearOperator(*bs, x1_.shape[-2], x2_.shape[-2])

        ls_exp = self.ls_0 + x1_[..., self.exp_par_ind].matmul(self.ls_v)
        ls = torch.exp(ls_exp)
        ls = ls[:,None]

        os_exp = self.os_0 + x1_[..., self.exp_par_ind].matmul(self.os_v)
        os = torch.exp(os_exp)

        active_dim = [i for i in range(x1_.shape[-1]) if not i in self.exp_par_ind]
        x1_ = x1_[...,active_dim] / ls
        x2_ = x2_[...,active_dim] / ls

        if torch.equal(x1, self.train_x): #Train-train covariance case

            if isinstance(self.data_size, int): # All blocks are the same size, we can emit a BlockDiagLinearOperator

                # Check that all control parameters match
                ns = [*bs, self.n_exp, self.data_size]
                xx = x1.clone().reshape([*ns, idim])[...,self.exp_par_ind]
                chk_diff = xx - xx[...,[0],:]
                chk_diff = chk_diff.to(bool)
                if torch.any(chk_diff):
                    raise ValueError("Not all training array blocks contain control parameters that all match.")
                
                os = os.reshape(ns)[...,:,0]
                os = os[...,None,None]
                x1_ = x1_.reshape([*ns,1])
                x2_ = x2_.reshape([*ns,1])

                batched_covar = self.base_kernel(x1_, x2_, diag=diag, **params).mul(os)
                covar = BlockDiagLinearOperator(batched_covar)
                if diag:
                    covar = covar.diag()

            else:
                n_pts = self.data_size.sum()
                covar = torch.zeros((*bs, n_pts, n_pts))
                i1 = 0
                for blk in range(self.n_exp):
                    i2 = i1 + self.data_size[blk]
                    # Check that all control parameters match
                    chk_diff = x1[...,i1:i2,self.exp_par_ind] - x1[...,i1,self.exp_par_ind]
                    chk_diff = chk_diff.to(bool)
                    if torch.any(chk_diff):
                        raise ValueError("Not all control parameters in training block {blk} are the same.")
                    
                    xx = x1_[..., i1:i2, :]
                    covar[..., i1:i2, i1:i2] = self.base_kernel(xx, xx, diag=diag, **params).mul(os[...,i1]).to_dense()
                    i1 = i2

        else: # Test-test covariance case
            # Check that all control parameters match
            chk_diff = x1[...,:,self.exp_par_ind] - x1[...,0,self.exp_par_ind]
            chk_diff = chk_diff.to(bool)
            if torch.any(chk_diff):
                raise ValueError("Not all control parameters of test array are the same.")

            covar = self.base_kernel(x1_, x2_, diag=diag, **params).mul(os[...,0])


        return covar

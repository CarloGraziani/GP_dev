from gpytorch.kernels import Kernel
import torch
from torch import Tensor
from linear_operator.operators.block_diag_linear_operator import BlockDiagLinearOperator
from linear_operator.operators.zero_linear_operator import ZeroLinearOperator
from copy import deepcopy

def fix_lengthscale(kernel_cls):
    """
    Ensure that a stationary kernel may be defined without it acquiring an adjustable lengthscale.
    """
    newcls = deepcopy(kernel_cls)
    newcls.is_stationary = True
    newcls.has_lengthscale = False
    newcls.lengthscale = 1.0
    return newcls

SKernel = fix_lengthscale(Kernel)
#SKernel.is_stationary = True

class ExperimentalUncertaintyKernel(SKernel):
    """
    """

#################
    def __init__(self,
                 base_kernel: Kernel,
                 data_size: int,
                 exp_par_size: tuple[int],
                 outputscale_fn: bool = False,
                 lengthscale_fn: bool = False,
                 **kwargs) -> None:

        super(ExperimentalUncertaintyKernel, self).__init__(**kwargs)

        self.is_stationary = base_kernel.is_stationary
        self.base_kernel = base_kernel
        self.data_size = data_size
        self.exp_par_size = exp_par_size
        epl = len(exp_par_size)

        self.register_parameter("os_0", 
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape,1)))

        self.register_parameter("ls_0", 
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape,1)))
        
        if outputscale_fn:
            self.register_parameter("os_v", 
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, epl)))
        else:
            self.os_v = torch.zeros(*self.batch_shape, epl)

        if lengthscale_fn:
            self.register_parameter("ls_v", 
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, epl)))
        else:
            self.ls_v = torch.zeros(*self.batch_shape, epl)

#################
    def forward(self, x1: Tensor, x2: Tensor, last_dim_is_batch: bool = False, 
                diag: bool = False, **params) -> Tensor:
        
        try:
            bs = torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2]) # Are batch shapes compatible?
        except RuntimeError:
            raise ValueError("Batch shapes of x1 and x2 are incompatible")

        x1_ = x1.clone() ; x2_ = x2.clone()
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -3)
            x2_ = x2_.transpose(-1, -3)

        if x1_.shape[-2] % self.data_size != 0 or x2_.shape[-2] % self.data_size != 0:
            raise ValueError("Input data size not a multiple of %d" % (self.data_size))
        
        if not x1_.shape[-1] == len(self.exp_par_size) + 1:
            raise ValueError("Only 1-D within-experiment data supported") # this may change someday
        
        if not torch.equal(x1_, x2_):
            return ZeroLinearOperator(tuple(*bs, x1_.shape[-2], x2_.shape[-2]))
        
        ns = [*x1_.shape[:-2], x1_.shape[-2] // self.data_size, self.data_size, x1_.shape[-1]]
        x1_ = x1_.reshape(ns)
        x2_ = x2_.reshape(ns)

        ls_exp = self.ls_0 + x1_[..., 0,self.exp_par_size].matmul(self.ls_v)
        ls = torch.exp(ls_exp)
        ls = ls[:,None,None]
        os_exp = self.os_0 + x1_[..., 0,self.exp_par_size].matmul(self.os_v)
        os = torch.exp(os_exp)
        os = os[:,None,None]

        active_dim = [i for i in range(x1_.shape[-1]) if not i in self.exp_par_size]
        x1_ = x1_[...,active_dim] / ls
        x2_ = x2_[...,active_dim] / ls

        batched_covar = self.base_kernel(x1_, x2_, diag=diag, **params).mul(os)
        covar = BlockDiagLinearOperator(batched_covar)
        if diag:
            covar = covar.diag()

        return covar

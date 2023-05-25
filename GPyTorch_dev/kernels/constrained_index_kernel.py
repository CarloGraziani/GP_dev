from typing import Callable

from torch import Tensor
from gpytorch.kernels import IndexKernel

######################################################################
class ConstrainedIndexKernel(IndexKernel):
    r"""
    A subclass of IndexKernel in which the B matrix and v vector are given
    by parametrized functions.

    Args:
        B_fn (callable):
            Function to be called returning the B matrix.
        v_fn (callable):
            Function to be called returning the v vector.
        batch_shape (torch.Size, optional):
            Set if the MultitaskKernel is operating on batches of data (and you want different
            parameters for each batch)

    Note that the rank and num_tasks parameters of the parent class are not required, as the
    B_fn and v_fn functions are expected to provide correctly-sized arrays. So we don't register
    any parameters or priors here, and we don't call IndexKernel's __init__().

    """

#################
    def __init__(self, 
                 B_fn: Callable[[], Tensor], 
                 v_fn: Callable[[], Tensor],
                 **kwargs) -> None:
        
        super(IndexKernel, self).__init__(**kwargs) # This should call Kernel's init
        self.B_fn = B_fn
        self.v_fn = v_fn

#################
    @property
    def var(self):
        return self.v_fn()
    
    @property
    def covar_factor(self):
        return self.B_fn()


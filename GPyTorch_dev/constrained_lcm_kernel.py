from typing import List
from constrained_multitask_kernel import ConstrainedMultitaskKernel
from gpytorch.kernels import LCMKernel, Kernel
import torch
from torch import Tensor
from torch.nn import ModuleList

######################################################################
class ConstrainedLCMKernel(LCMKernel):
    r"""
    A subclass of LCMKernel in which  all the B matrices and v vectors are given
    by parametrized functions, and in which the MultitaskKernel instances are replaced by
    instances of ConstrainedMultitaskKernel.

    Note that the rank and num_tasks parameters are redundantly specified,
    explicitly in the argument list and implicitly through the shapes of the Tensors returned by
    the B_fn and v_fn of the ConstrainedMultitaskKernel instances. We don't call LCMKernel's __init__() 
    as a consequence.

    Args:
    cmt_kernels (:type: list of `ConstrainedMultitaskKernel` objects): A list of ConstrainedMultitaskKernel
    instances.
    
    """

#################
    def __init__(self,
                 cmt_kernels: List[ConstrainedMultitaskKernel],
                 **kwargs) -> None:
        
        if len(cmt_kernels) < 1:
            raise ValueError("At least one base kernel must be provided.")
        for ind, k in enumerate(cmt_kernels):
            if not isinstance(k, ConstrainedMultitaskKernel):
                raise ValueError(f" Entry # {ind} of cmt_kernels is not a ConstrainedMultitaskKernel object")
            if ind == 0:
                num_tasks = k.num_tasks
            elif k.num_tasks != num_tasks:
                raise ValueError(f"ConstrainedMultitaskKernel entry # {ind} has inconsistent num_tasks")

        super(LCMKernel, self).__init__(**kwargs) # Should call Kernel's __init__()

        self.covar_module_list = ModuleList(cmt_kernels)
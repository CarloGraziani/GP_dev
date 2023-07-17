from typing import Callable
from constrained_index_kernel import ConstrainedIndexKernel
from gpytorch.kernels import MultitaskKernel, Kernel
from torch import Tensor

######################################################################
class ConstrainedMultitaskKernel(MultitaskKernel):
    r"""
    A subclass of MultitaskKernel in which the B matrix and v vector are given
    by parametrized functions, and in which the IndexKernel is replaced by an
    instance of ConstrainedIndexKernel.

    Args:
    :param ~gpytorch.kernels.Kernel data_covar_module: Kernel to use as the data kernel.
    :param int num_tasks: Number of tasks
    :param int rank: (default 1) Rank of index kernel to use for task covariance matrix.
    :param Callable[[], Tensor((num_tasks, rank))] B_fn: Callable returning the (num_tasks x rank) B matrix
    :param Callable[[], Tensor((num_tasks,))] v_fn: Callable returning the num_tasks-dimensional v vector

    Note that the rank and num_tasks parameters are redundantly specified,
    explicitly in the argument list and implicitly through the shapes of the Tensors returned by
    B_fn and v_fn. We don't call MultitaskKernel's __init__() as a consequence.

    """

#################
    def __init__(self,
                 data_covar_module: Kernel,
                 num_tasks: int,
                 B_fn: Callable[[], Tensor],
                 v_fn: Callable[[], Tensor],
                 **kwargs) -> None:
        
        super(MultitaskKernel, self).__init__(**kwargs) # This is Kernel's __init_()

        s = v_fn().shape
        if s[-1] != num_tasks:
            raise ValueError("v_fn has inconsistent num_tasks")
        s = B_fn().shape
        if s[-2] != num_tasks:
            raise ValueError("B_fn has inconsistent rank")
        
        self.task_covar_module = ConstrainedIndexKernel(B_fn, v_fn, **kwargs)
        self.data_covar_module = data_covar_module
        self.num_tasks = num_tasks

#################
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        self.task_covar_module.input = (x1, x2)
        res = super().forward(x1, x2, diag, last_dim_is_batch, **params)
        return res
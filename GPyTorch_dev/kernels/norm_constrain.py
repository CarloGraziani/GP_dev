from typing import Any, Optional

import torch
from torch import Tensor
from torch.special import gammaln
from torch.nn import ModuleList
from math import sqrt

from gpytorch import Module
from gpytorch.kernels import Kernel
from gpytorch.means import Mean
from gpytorch.constraints import Positive, GreaterThan


######################################################################
class Norm2ConstrainedKernel(Kernel):
    r"""
    Kernel for a GP intended to sample only functions :math:`f(x)` that satisfy
    an :math:`x^2`-weighted integral normalization condition, that is

    .. math::
        \int_0^\infty dx\,x^2f(x)=N

    This kernel is built upon a non-stationary base kernel :math:`K(x_1,x_2)`,
    which then has its value along the normalization integral projected away to
    yield the actual kernel :math:`C(x_1,x_2)`. That is to say,

    .. math::
        C(x_1,x_2)=K(x_1,x_2)-C_1(x_1)C_1(x_2)/C_0

    where

    .. math::

    \begin{align}
        C_1(x)\equiv& \int_0^\infty dx_1\,x_1^2\times K(x,x_1) \\
        C_2(x)\equiv \int_0^\infty dx\, C_1(x)
    \end{align}

    This form may be derived by treating the normalization condition (a linear
    function on the GP) as an observation made from a prior zero-mean GP with
    kernel :math:`K(x_1,x_2)`, and conditioning on it. The resulting GP has
    this kernel, and a mean function given by

    .. math::
        \mu(x)=(C_1(x)/C_0) * N

    The corresponding mean function is completely determined by the kernel and
    its hyperparameters. So we instantiate this class and the mean class
    together from the container class that subclasses
    Norm2Constrained_Container, which is in charge of registering parameters,
    and passing them to this instance through the arguments to __init__ of both
    mean and covariance. In other words, it's probably a bad idea to create a
    stand-alone instance of this class.

    Note that this is a 1-d model only. Also, it is non-stationary.

    :param base_kernel_eval: Function that computes the base kernel
    :type base_kernel_eval: callable
    :param C1: Function that computes the integral of the base kernel over one of its arguments.
    :type C1: callable
    :param C0: Function that computes the integral of the base kernel over both of its arguments.
    :type C0: callable

    """

    has_lengthscale = False

#################
    def __init__(self, base_kernel_eval, C1, C0, **kwargs):

        super(Norm2ConstrainedKernel, self).__init__(**kwargs)

        self.base_kernel_eval = base_kernel_eval
        self.C1 = C1
        self.C0 = C0


#################
    def forward(self, x1: Tensor, x2: Tensor, last_dim_is_batch: bool = False, diag: bool = False, **params) -> Tensor:

        x1_ = x1.clone() ; x2_ = x2.clone()
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2)
            x2_ = x2_.transpose(-1, -2)
        x1_ = x1_.squeeze(-1) ; x2_ = x2_.squeeze(-1) # this is a 1-d kernel
        
        c0 = self.C0()
        if diag:
            c1_1 = self.C1(x1_)
            res = self.base_kernel_eval(x1_, x1_) - c1_1**2/c0
        else:
            x1_ = x1_.unsqueeze(-1)
            x2_ = x2_.unsqueeze(-2)
            c1_1 = self.C1(x1_)
            c1_2 = self.C1(x2_)
            res = self.base_kernel_eval(x1_, x2_) - c1_1*c1_2 / c0
        
        return res

######################################################################
class Norm2ConstrainedMean(Mean):
    r"""
    A mean intended to accompany a NormConstrainedKernel, and to be initialized
    using that kernel's parameters.

    If the base kernel is :math:`K(x_1,x_2)`, then this mean function is

    .. math::

        \begin{align}
        \mu(x)=&(C_1(x)/C_0) * N \\
        C_1(x)\equiv& \int_0^\infty dx_1\,x_1^2\times K(x,x_1) \\
        C_2(x)\equiv \int_0^\infty dx\, C_1(x)
        \end{align}
    
    where :math:`N` is the desired integral normalization of :math:`\mu(x)` and
    of all the functions sampled by this process, weighted by :math:`x^2`.

    :param C1: Function that computes the integral of the base kernel over one of its arguments.
    :type C1: callable
    :param C0: Function that computes the integral of the base kernel over both of its arguments.
    :type C0: callable
    :param norm_val: (Default: 1.0) Desired integral normalization.
    :type norm_val: float, optional

    This mean function is completely determined by the corresponding kernel and
    its hyperparameters. So we instantiate this class and the kernel class
    together from the container class that subclasses
    Norm2Constrained_Container, which is in charge of registering parameters,
    and passing them to this instance through the arguments to __init__ of both
    mean and covariance. In other words, it's probably a bad idea to create a
    stand-alone instance of this class.

    This mean has no independent learned parameters. The corresponding
    parameters from the accompanying kernel model should be assigned as
    attributes.

    """
#################
    def __init__(self, C1, C0, norm_val=1.0, **kwargs):
        
        super(Norm2ConstrainedMean, self).__init__()

        self.C1 = C1
        self.C0 = C0
        self.norm_val = norm_val

#################
    def forward(self, input):

        x = input.squeeze(-1)
        c1 = self.C1(x)
        c0 = self.C0()
        res = c1 * self.norm_val / c0
        return res    

######################################################################
class Norm2ConstrainedContainer(Module):
    r"""
    Class for instantiating a Norm2ConstrainedKernel and a
    Norm2ConstrainedMean, and sharing the relevant parameters with them through
    the closed-form functions base_kernel_eval, C1, and C0, which are the base
    kernel, its integral over one argument, and its integral over both
    arguments, respectively. Subclass this and overwrite base_kernel, C1, and
    C0 with appropriate functions.

    :param norm_val: (Default: 1.0) Desired integral normalization.
    :type norm_val: float, optional

    """

    def __init__(self, norm_val=1.0, batch_shape = None, **kwargs):

        super(Norm2ConstrainedContainer, self).__init__(**kwargs)

        self.batch_shape = torch.Size([]) if batch_shape is None else batch_shape
        self.covar_module = Norm2ConstrainedKernel(self.base_kernel_eval, self.C1, self.C0)
        self.mean_module = Norm2ConstrainedMean(self.C1, self.C0, norm_val)


#################
    def base_kernel_eval(self, z1, z2):
        
        raise NotImplementedError()
    

#################
    def C1(self, x):
        
        raise NotImplementedError()


#################
    def C0(self):
        
        raise NotImplementedError()


######################################################################
class Norm2ConstrainedContainer_SE(Norm2ConstrainedContainer):
    r"""
    A Norm2-constrained model based on a tapered SE base kernel. That is, the
    base kernel is :math:`K(x_1,x_2)=A \times \exp(-(x_1^2-x_2^2)/\gamma^2 -
    (x_1-x_2)^2/\sigma^2)`

    :param norm_val: (Default: 1.0) Desired integral normalization.
    :type norm_val: float, optional
    """

    def __init__(self, norm_val=1.0, **kwargs):
        super(Norm2ConstrainedContainer_SE, self).__init__(norm_val, **kwargs)

        self.register_parameter("raw_sigma", 
                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        sigma_constraint = Positive()
        self.register_constraint("raw_sigma", sigma_constraint)

        self.register_parameter("raw_gamma", 
                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        gamma_constraint = Positive()
        self.register_constraint("raw_gamma", gamma_constraint)

        self.register_parameter("raw_A", 
                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        A_constraint = Positive()
        self.register_constraint("raw_A", A_constraint)

#################
    @property
    def sigma(self):
        return self.raw_sigma_constraint.transform(self.raw_sigma)

    @sigma.setter
    def sigma(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_gamma)
        self.initialize(raw_sigma=self.raw_sigma_constraint.inverse_transform(value))

    @property
    def gamma(self):
        return self.raw_gamma_constraint.transform(self.raw_gamma)

    @gamma.setter
    def gamma(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_gamma)
        self.initialize(raw_gamma=self.raw_gamma_constraint.inverse_transform(value))

    @property
    def A(self):
        return self.raw_A_constraint.transform(self.raw_A)

    @A.setter
    def A(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_A)
        self.initialize(raw_A=self.raw_A_constraint.inverse_transform(value))


#################
    def base_kernel_eval(self, z1, z2):
        qf = (z1-z2)**2 / self.sigma**2 + (z1**2 + z2**2) / self.gamma**2
        res = self.A * torch.exp(-qf)
        return res
    

#################
    def C1(self, x):
        a = self.sigma**-2 + self.gamma**-2
        b = -self.sigma**-2
        sa = a.sqrt()
        res = ( sqrt(torch.pi) / (4*sa**3) ) * (1 + 2 * (b*x)**2/a) \
            * torch.exp(-(a - b**2/a) * x**2) * torch.erfc(b*x/sa)
        res = res - (b*x/(2*a**2)) * torch.exp(-a*x**2)
        res = res * self.A

        return res

#################
    def C0(self):
        a = self.sigma**-2 + self.gamma**-2
        b = -self.sigma**-2
        amb = a**2 - b**2

        res = (1/8) * amb**-2.5 * (a**2 + 2*b**2) * \
              (torch.pi/2 - torch.arcsin(b/a))
        res = res - (3/8) * b / amb**2
        res = res * self.A

        return res
    
######################################################################
class Norm2ConstrainedContainer_rational(Norm2ConstrainedContainer):
    r"""
    A Norm2-constrained model based on a rational base kernel of the form
    ::math::`K(x_1,x_2)=A/(x_1+x_2+\alpha)^p`, with :math:`p>6.0` to ensure
    that the :math:`x^2`-weighted normalization is finite.
    
    :param norm_val: (Default: 1.0) Desired integral normalization.
    :type norm_val: float, optional
    """

    def __init__(self, norm_val, **kwargs):
        super(Norm2ConstrainedContainer_rational, self).__init__(norm_val, **kwargs)

        self.register_parameter("raw_alpha", 
                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        alpha_constraint = Positive()
        self.register_constraint("raw_alpha", alpha_constraint)

        self.register_parameter("raw_p", 
                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        p_constraint = GreaterThan(6.0)
        self.register_constraint("raw_p", p_constraint)

        self.register_parameter("raw_A", 
                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        A_constraint = Positive()
        self.register_constraint("raw_A", A_constraint)

#################
    @property
    def alpha(self):
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_alpha)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    @property
    def p(self):
        return self.raw_p_constraint.transform(self.raw_p)
    
    @p.setter
    def p(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_p)
        self.initialize(raw_p=self.raw_p_constraint.inverse_transform(value))

    @property
    def A(self):
        return self.raw_A_constraint.transform(self.raw_A)

    @A.setter
    def A(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_A)
        self.initialize(raw_A=self.raw_A_constraint.inverse_transform(value))

#################
    def base_kernel_eval(self, z1, z2):

        ret = self.A * (z1 + z2 + self.alpha)**-self.p
        return ret
    

#################
    def C1(self, x):

        coef_1 = 2 * torch.exp(gammaln(self.p-3) - gammaln(self.p))
        ret = self.A * coef_1 * (x + self.alpha)**-(self.p-3)
        return ret

#################
    def C0(self):

        coef_0 = 4 * torch.exp(gammaln(self.p-6) - gammaln(self.p))
        ret = self.A * coef_0 * (self.alpha)**-(self.p-6)
        return ret

######################################################################
class SimplexConstraint(Module):
    r"""
    Constraint that forces an n-parameter vector c to live on an n+1-simplex,
    so that torch.all(c>=0) is True and c.sum()<1.0. The constraint operates by
    exploiting Aitchison-style centered log variable as raw variables.

    """

    enforced = True

#################
    def __init__(self, initial_value=None):
        super(SimplexConstraint, self).__init__()

        if initial_value is not None:
            self._initial_value = self.inverse_transform(torch.as_tensor(initial_value))
        else:
            self._initial_value = None


#################

    def transform(self, z):
        r"""
        Go from centered-log to simplex
        """
        znp1 = -z.sum(dim=-1,keepdim=True)
        c = torch.cat((z,znp1)) # n+1-dimensional, sums to zero
        c = c - c.max() # We're about to exponentiate tmp, scale factor is irrelevant
        c = c.exp()
        c = c / c.sum(dim=-1) # Sums to 1, positive, ergo on n+1-simplex

        return c[...,:-1] # back to n-dimensional
    
#################
    def inverse_transform(self, c):
        r"""
        Go from simplex to centered-log
        """

        if not self.check(c):
            raise ValueError("Invalid simplex parameters")
        
        cnp1 = 1 - c.sum(dim=-1, keepdim=True)
        z = torch.cat((c,cnp1)) # n+1-dimensional, sums to one
        z = z.log()
        z = z - z.mean(dim=-1) # sums to zero, all parameters range is [-inf,inf]

        return z[...,:-1] # back to n-dimensional
    
#################
    @property
    def initial_value(self) -> Optional[Tensor]:
        """
        The initial parameter value (if specified, None otherwise)
        """
        return self._initial_value

#################
    def check(self, c):

        return bool(torch.all(c>=0) and c.sum(dim=-1) <= 1.0)
    
#################
    def check_raw(self, z):

# Technically, this should always be True unless math is wrong, or z has bad dtype, maybe.
        return self.check(self.transform(z)) 


######################################################################
class Norm2ConstrainedContainer_ConvexCombination(Norm2ConstrainedContainer):
    r"""
    A Norm2-constrained model based on a base kernel that is a convex combination of norm2-constrained kernels.

    :param kernels: List or tuple of Norm2ConstrainedContainer-encapsulated models

    :param norm_val: (Default: 1.0) Desired integral normalization.
    :type norm_val: float, optional

    """

#################
    def __init__(self, kernels, norm_val=1.0, **kwargs):

        super(Norm2ConstrainedContainer_ConvexCombination, self).__init__(norm_val, **kwargs)

        for kernel in kernels:
            if not isinstance(kernel, Norm2ConstrainedContainer):
                raise ValueError("Subkernel not a Norm2ConstrainedContainer instance")
            if kernel.mean_module.norm_val != norm_val:
                raise ValueError("Inconsistent normalizations")

        self.kernels = ModuleList(kernels)
        
        nkernels = len(kernels)
        self.register_parameter("raw_compositions", 
                            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, nkernels-1)))
        comp_constraint = SimplexConstraint()
        self.register_constraint("raw_compositions", comp_constraint)


#################
    @property
    def compositions(self):
        return self.raw_compositions_constraint.transform(self.raw_compositions)

    @compositions.setter
    def compositions(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_compositions)
        self.initialize(raw_compositions=self.raw_compositions_constraint.inverse_transform(value))

    @property
    def full_compositions(self):
        c = self.raw_compositions_constraint.transform(self.raw_compositions)
        c = torch.cat((c, 1 - c.sum(dim=-1, keepdims=True)))
        return c


#################
    def base_kernel_eval(self, z1, z2):

        ret = 0
        for ik, kernel in enumerate(self.kernels):
            ret = ret + kernel.base_kernel_eval(z1, z2) * self.full_compositions[ik]
        return ret
    

#################
    def C1(self, x):

        ret = 0
        for ik, kernel in enumerate(self.kernels):
            ret = ret + kernel.C1(x) * self.full_compositions[ik]
        return ret

#################
    def C0(self):

        ret = 0
        for ik, kernel in enumerate(self.kernels):
            ret = ret + kernel.C0() * self.full_compositions[ik]
        return ret
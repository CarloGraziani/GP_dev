from typing import Optional
import torch
from torch import Tensor
from gpytorch import Module


######################################################################
class SimplexConstraint(Module):
    r"""
    Constraint that forces an n-parameter vector c to live on an n+1-simplex, so that torch.all(c>=0) is True and c.sum()<1.0.  The constraint operates by exploiting Aitchison-style centered log variable as raw variables.

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


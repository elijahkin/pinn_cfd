"""Classical finite-volume solvers for the 1-D convection-diffusion equation.

- fv1d        : explicit methods for pure convection
- convdiff1d  : implicit methods for convection-diffusion
- limiters    : minmod and van Leer MC flux limiters
- reference   : high-accuracy reference solutions
"""

from . import convdiff1d, fv1d, limiters, reference

__all__ = ["fv1d", "convdiff1d", "limiters", "reference"]

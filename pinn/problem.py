"""Problem definition for the 1-D convection-diffusion equation.

    u_t + f_x = nu u_xx,   f = a u (linear)  or  f = u^2 / 2 (nonlinear)

with a discontinuous initial condition, an oscillatory Dirichlet condition at
x = 0, and a non-reflective (zero-gradient) condition at x = pi.
"""

import math
from dataclasses import dataclass, field

import torch


@dataclass
class ConvectionDiffusionProblem:
    kind: str = "linear"
    nu: float = 0.012
    a: float = 1.0
    x_min: float = 0.0
    x_max: float = math.pi
    t_min: float = 0.0
    t_max: float = math.pi
    bc_omegas: tuple = (4.0, 8.0)
    ic_eps: float = 0.0

    def __post_init__(self):
        if self.kind not in ("linear", "nonlinear"):
            raise ValueError(f"kind must be 'linear' or 'nonlinear', got {self.kind!r}")
        if self.ic_eps < 0.0:
            raise ValueError("ic_eps must be non-negative")

    @property
    def domain(self):
        return (self.t_min, self.t_max, self.x_min, self.x_max)

    def ic(self, x):
        """Initial condition. ic_eps > 0 replaces the jump with a tanh layer."""
        if self.ic_eps > 0.0:
            return 1.0 + 0.5 * torch.tanh((x - 0.5 * math.pi) / self.ic_eps)
        half = torch.full_like(x, 0.5)
        three_halves = torch.full_like(x, 1.5)
        return torch.where(x < 0.5 * math.pi, half, three_halves)

    def left_bc(self, t):
        """Oscillatory Dirichlet boundary condition at x = 0."""
        out = torch.ones_like(t)
        for w in self.bc_omegas:
            out = out - 0.25 * torch.cos(w * t)
        return out

    def flux(self, u):
        if self.kind == "linear":
            return self.a * u
        return 0.5 * u * u

    def residual(self, u, u_t, u_x, u_xx):
        """PDE residual u_t + f_x - nu u_xx."""
        f_x = self.a * u_x if self.kind == "linear" else u * u_x
        return u_t + f_x - self.nu * u_xx

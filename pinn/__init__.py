"""Physics-informed neural network solvers for convection-diffusion.

- problem : PDE/IC/BC definition
- model   : the MLP (with optional Fourier features)
- data    : collocation-point sampling
- losses  : convection-diffusion residual loss
- train   : training driver
"""

from .losses import convection_diffusion_loss
from .model import Pinn
from .problem import ConvectionDiffusionProblem
from .train import (
    TrainConfig,
    device_description,
    predict_grid,
    resolve_device,
    train_pinn,
)

__all__ = [
    "Pinn",
    "ConvectionDiffusionProblem",
    "TrainConfig",
    "convection_diffusion_loss",
    "device_description",
    "train_pinn",
    "predict_grid",
    "resolve_device",
]

"""Training driver for the convection-diffusion PINN.

Wires together the model, the sampled collocation points, and the loss, using
Adam with a MultiStepLR decay and weighted PDE/IC/BC losses.
"""

from dataclasses import dataclass, field

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from .data import generate_training_data
from .losses import convection_diffusion_loss
from .model import Pinn


@dataclass
class TrainConfig:
    num_epochs: int = 3000
    hidden_size: int = 50
    num_hidden_layers: int = 4
    activation: str = "tanh"
    fourier_features: int = 0
    fourier_sigma: float = 1.0
    arch: str = "mlp"
    hard_ic: bool = False
    adaptive_activation: bool = False
    activation_scale: float = 1.0
    siren_omega: float = 3.0
    lr: float = 1e-3
    N_IC: int = 400
    N_BC: int = 400
    N_PDE: int = 20000
    batch_size: int = None
    loss_weights: dict = field(
        default_factory=lambda: {"pde": 1.0, "ic": 10.0, "bc": 1.0}
    )
    weighting: str = "fixed"
    weight_update_every: int = 100
    weight_alpha: float = 0.1
    milestones: tuple = (500, 1000, 1500)
    gamma: float = 0.5
    log_every: int = 100
    seed: int = 0
    device: str = None


def resolve_device(requested=None):
    """Return the torch device string, defaulting to CUDA when available."""
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def device_description(device):
    """Human-readable name for `device`, e.g. 'cuda (NVIDIA GeForce RTX 3060)'."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return f"{device} ({torch.cuda.get_device_name(0)})"
    return str(device)


def train_pinn(problem, config=None, verbose=True, callback=None):
    """Train a PINN for `problem`. Returns (model, data, config)."""
    if config is None:
        config = TrainConfig()
    device = resolve_device(config.device)
    config.device = device
    if verbose:
        print(f"[pinn] training on {device_description(device)}")
    torch.manual_seed(config.seed)

    data = generate_training_data(
        config.N_IC, config.N_BC, config.N_PDE, problem, device=device, seed=config.seed
    )

    model = Pinn(
        in_size=2,
        hidden_size=config.hidden_size,
        out_size=1,
        num_hidden_layers=config.num_hidden_layers,
        activation=config.activation,
        fourier_features=config.fourier_features,
        fourier_sigma=config.fourier_sigma,
        seed=config.seed,
        arch=config.arch,
        hard_ic=config.hard_ic,
        adaptive_activation=config.adaptive_activation,
        activation_scale=config.activation_scale,
        siren_omega=config.siren_omega,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=config.lr)
    scheduler = MultiStepLR(
        optimizer, milestones=list(config.milestones), gamma=config.gamma
    )

    batch_size = config.batch_size or max(1, config.N_PDE // 64)

    model.fit(
        num_epochs=config.num_epochs,
        loss_fn=convection_diffusion_loss,
        pde_data=data["pde"],
        batch_size=batch_size,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_weights=config.loss_weights,
        weighting=config.weighting,
        weight_update_every=config.weight_update_every,
        weight_alpha=config.weight_alpha,
        log_every=config.log_every,
        callback=callback,
        verbose=verbose,
        ic_data=data["ic"],
        bc_left_data=data["bc_left"],
        bc_right_data=data["bc_right"],
        base_loss_fn=torch.nn.MSELoss(),
        problem=problem,
    )
    return model, data, config


def predict_grid(model, problem, x, t):
    """Evaluate the network on a (t, x) grid. x, t are 1-D numpy arrays."""
    import numpy as np

    T, X = np.meshgrid(t, x, indexing="ij")
    pts = np.stack((T.ravel(), X.ravel()), axis=1)
    model.eval()
    with torch.no_grad():
        U = model(
            torch.from_numpy(pts).float().to(next(model.parameters()).device)
        ).squeeze(-1)
    return U.cpu().numpy().reshape(T.shape)

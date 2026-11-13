"""Validation of the convection-diffusion PINN.

Run from the project root:

    python -m pinn_cfd.experiments.validate_pinn

Checks:
  A. Data shapes, device, and requires_grad.
  B. The PDE residual loss is ~0 for a manufactured exact solution of the
     linear convection-diffusion equation.
  C. Gradients flow to the network parameters.
  D. A short smoke training run reduces the IC loss and stays finite.
"""

import numpy as np
import torch

from pinn_cfd.pinn import (
    ConvectionDiffusionProblem,
    Pinn,
    TrainConfig,
    convection_diffusion_loss,
    predict_grid,
    train_pinn,
)
from pinn_cfd.pinn.data import generate_training_data

PI = np.pi


class ExactLinear(torch.nn.Module):
    """u(x,t) = exp(-nu k^2 t) cos(k (x - a t)), an exact solution."""

    def __init__(self, nu, a, k=1.0):
        super().__init__()
        self.nu, self.a, self.k = nu, a, k

    def forward(self, tx):
        t, x = tx[:, 0:1], tx[:, 1:2]
        return torch.exp(-self.nu * self.k**2 * t) * torch.cos(
            self.k * (x - self.a * t)
        )


def check_data():
    print("A. Data generation")
    problem = ConvectionDiffusionProblem()
    data = generate_training_data(50, 40, 500, problem, seed=1)
    ok = True
    for key, n in (("ic", 50), ("bc_left", 40), ("bc_right", 40), ("pde", 500)):
        tx = data[key]
        good = (
            tuple(tx.shape) == (n, 2) and tx.requires_grad and tx.dtype == torch.float32
        )
        ok = ok and good
        print(
            f"   {key:<9}: shape={tuple(tx.shape)} "
            f"requires_grad={tx.requires_grad}  {'ok' if good else 'BAD'}"
        )
    return ok


def check_residual():
    print("\nB. Manufactured-solution PDE residual")
    nu, a, k = 0.05, 1.0, 2.0
    problem = ConvectionDiffusionProblem(kind="linear", nu=nu, a=a)
    data = generate_training_data(20, 20, 4000, problem, seed=2)
    exact = ExactLinear(nu, a, k)
    losses = convection_diffusion_loss(
        exact,
        data["pde"],
        data["ic"],
        data["bc_left"],
        data["bc_right"],
        torch.nn.MSELoss(),
        problem,
    )
    pde = losses["pde"].item()
    print(f"   MSE of residual = {pde:.3e}")
    return pde < 1e-8


def check_gradients():
    print("\nC. Gradient flow")
    problem = ConvectionDiffusionProblem()
    data = generate_training_data(20, 20, 1000, problem, seed=3)
    model = Pinn(hidden_size=16, num_hidden_layers=2)
    losses = convection_diffusion_loss(
        model,
        data["pde"],
        data["ic"],
        data["bc_left"],
        data["bc_right"],
        torch.nn.MSELoss(),
        problem,
    )
    total = losses["pde"] + losses["ic"] + losses["bc"]
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    finite = all(torch.isfinite(g).all() for g in grads)
    nonzero = any(g.abs().sum() > 0 for g in grads)
    print(f"   {len(grads)} parameter tensors, finite={finite}, " f"nonzero={nonzero}")
    return finite and nonzero


def check_smoke_train():
    print("\nD. Smoke training (100 epochs)")
    problem = ConvectionDiffusionProblem(kind="linear", nu=0.012)
    config = TrainConfig(
        num_epochs=100,
        N_PDE=1000,
        N_IC=50,
        N_BC=50,
        hidden_size=32,
        num_hidden_layers=3,
        log_every=50,
        seed=4,
    )
    model, data, _ = train_pinn(problem, config, verbose=False)

    x = np.linspace(0.0, PI, 50)
    t = np.linspace(0.0, PI, 10)
    U = predict_grid(model, problem, x, t)
    finite = np.all(np.isfinite(U))
    ic0, ic1 = model.loss_ic_history[0], model.loss_ic_history[-1]
    pde1 = model.loss_pde_history[-1]
    print(f"   IC loss: {ic0:.3e} -> {ic1:.3e} (reduced={ic1 < ic0})")
    print(f"   final PDE loss = {pde1:.3e}, prediction finite={finite}")
    return finite and ic1 < ic0


def main():
    ok = check_data()
    ok = check_residual() and ok
    ok = check_gradients() and ok
    ok = check_smoke_train() and ok
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

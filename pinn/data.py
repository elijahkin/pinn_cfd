"""Training-point generation for the convection-diffusion PINN.

Points are generated directly in torch so they can be placed on the target
device, and are stored as (t, x) columns to match the network input
convention.
"""

import torch


def _uniform(lo, hi, n, generator):
    return lo + (hi - lo) * torch.rand(n, generator=generator)


def generate_training_data(
    N_IC, N_BC, N_PDE, problem, device="cpu", seed=None, visualize=False
):
    """Sample IC, boundary, and PDE collocation points.

    Returns a dict with keys 'ic', 'bc_left', 'bc_right', 'pde', each a
    (N, 2) float tensor of [t, x] points with requires_grad enabled.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    x_min, x_max = problem.x_min, problem.x_max
    t_min, t_max = problem.t_min, problem.t_max

    # Initial condition: t = t_min, x uniform.
    x_ic = _uniform(x_min, x_max, N_IC, generator)
    t_ic = torch.full((N_IC,), t_min)

    # Boundary conditions: x at the ends, t uniform.
    t_bc = _uniform(t_min, t_max, N_BC, generator)
    x_left = torch.full((N_BC,), x_min)
    x_right = torch.full((N_BC,), x_max)

    # PDE collocation points: uniform in the space-time domain.
    t_pde = _uniform(t_min, t_max, N_PDE, generator)
    x_pde = _uniform(x_min, x_max, N_PDE, generator)

    def make(t, x):
        tx = torch.stack((t, x), dim=1).float().to(device)
        return tx.requires_grad_(True)

    data = {
        "ic": make(t_ic, x_ic),
        "bc_left": make(t_bc, x_left),
        "bc_right": make(t_bc, x_right),
        "pde": make(t_pde, x_pde),
    }

    if visualize:
        _visualize(data)

    return data


def _visualize(data):
    import matplotlib.pyplot as plt

    ic, bl, br, pde = data["ic"], data["bc_left"], data["bc_right"], data["pde"]
    plt.scatter(pde[:, 0].cpu(), pde[:, 1].cpu(), s=0.5, label="PDE")
    plt.scatter(ic[:, 0].cpu(), ic[:, 1].cpu(), s=8, label="IC")
    plt.scatter(bl[:, 0].cpu(), bl[:, 1].cpu(), s=8, marker="^", label="BC")
    plt.scatter(br[:, 0].cpu(), br[:, 1].cpu(), s=8, marker="^")
    plt.xlabel("$t$")
    plt.ylabel("$x$")
    plt.title("Training points")
    plt.legend()
    plt.show()

"""Loss function for the convection-diffusion PINN.

All point sets are evaluated in a single forward pass and a single first-order
autograd call, with a second autograd call only for the u_xx term on the PDE
points.
"""

import torch


def convection_diffusion_loss(
    model, pde_batch, ic_data, bc_left_data, bc_right_data, base_loss_fn, problem
):
    n_pde = pde_batch.shape[0]
    n_ic = ic_data.shape[0]
    n_bc = bc_left_data.shape[0]

    X = torch.cat((pde_batch, ic_data, bc_left_data, bc_right_data), dim=0)
    U = model(X).squeeze(-1)

    dU = torch.autograd.grad(U, X, grad_outputs=torch.ones_like(U), create_graph=True)[
        0
    ]

    i1 = n_pde
    i2 = n_pde + n_ic
    i3 = n_pde + n_ic + n_bc
    U_pde = U[:i1]
    # The IC penalty acts on the unconstrained network. With a hard IC, the
    # constrained output is the exact datum at t=0 and gives no gradient.
    raw_model = getattr(model, "raw", model)
    U_IC = raw_model(ic_data).squeeze(-1)
    U_left = U[i2:i3]
    U_right = U[i3:]
    U_t = dU[:i1, 0]
    U_x = dU[:i1, 1]
    Ux_right = dU[i3:, 1]

    # Second derivative is only needed on the interior points.
    U_xx = torch.autograd.grad(
        U_x, pde_batch, grad_outputs=torch.ones_like(U_x), create_graph=True
    )[0][:, 1]

    residual = problem.residual(U_pde, U_t, U_x, U_xx)
    loss_pde = base_loss_fn(residual, torch.zeros_like(residual))

    loss_ic = base_loss_fn(U_IC, problem.ic(ic_data[:, 1]))

    # Dirichlet at x=0 and zero-gradient (non-reflective) at x=pi.
    loss_bc = base_loss_fn(U_left, problem.left_bc(bc_left_data[:, 0]))
    loss_bc = loss_bc + base_loss_fn(Ux_right, torch.zeros_like(Ux_right))

    return {"pde": loss_pde, "ic": loss_ic, "bc": loss_bc}

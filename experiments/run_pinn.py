"""Train a convection-diffusion PINN and score it against the reference.

Example (short smoke run):

    python -m pinn_cfd.experiments.run_pinn --epochs 150 --N-pde 3000 \
        --hidden 40 --layers 3 --nu 0.012 --out results/linear_012

Writes an npz with the fields/history and PNGs of the solution and error.
"""

import argparse
import os
import time

import numpy as np

from pinn_cfd.classic import reference
from pinn_cfd.classic.fv1d import l1_error
from pinn_cfd.pinn import (
    ConvectionDiffusionProblem,
    TrainConfig,
    device_description,
    predict_grid,
    resolve_device,
    train_pinn,
)

PI = np.pi


def numpy_boundary(omegas):
    def bc(t):
        out = np.ones_like(np.asarray(t, dtype=float))
        for w in omegas:
            out = out - 0.25 * np.cos(w * t)
        return out

    return bc


def numpy_ic(eps):
    if eps <= 0.0:
        return lambda x: np.where(np.asarray(x) < 0.5 * PI, 0.5, 1.5)
    return lambda x: 1.0 + 0.5 * np.tanh((np.asarray(x) - 0.5 * PI) / eps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", default="linear", choices=["linear", "nonlinear"])
    p.add_argument("--nu", type=float, default=0.012)
    p.add_argument("--t-end", type=float, default=PI)
    p.add_argument("--ic-eps", type=float, default=0.0)
    p.add_argument(
        "--omegas",
        type=float,
        nargs="+",
        default=[4.0, 8.0],
        help="frequencies in the Dirichlet BC (spectral-bias study)",
    )
    p.add_argument("--device", default=None, help="torch device, e.g. 'cuda'")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--N-pde", type=int, default=20000)
    p.add_argument("--N-ic", type=int, default=400)
    p.add_argument("--N-bc", type=int, default=400)
    p.add_argument("--hidden", type=int, default=50)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--activation", default="tanh")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--milestones", type=int, nargs="+", default=[1200, 2200])
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--w-pde", type=float, default=1.0)
    p.add_argument("--w-ic", type=float, default=10.0)
    p.add_argument("--w-bc", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nx", type=int, default=201)
    p.add_argument("--nt", type=int, default=101)
    p.add_argument("--ref-dx", type=float, default=PI / 800)
    p.add_argument("--out", default="pinn_cfd/results/pinn")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    problem = ConvectionDiffusionProblem(
        kind=args.kind,
        nu=args.nu,
        t_max=args.t_end,
        ic_eps=args.ic_eps,
        bc_omegas=tuple(args.omegas),
    )
    device = resolve_device(args.device)
    config = TrainConfig(
        num_epochs=args.epochs,
        N_PDE=args.N_pde,
        N_IC=args.N_ic,
        N_BC=args.N_bc,
        hidden_size=args.hidden,
        num_hidden_layers=args.layers,
        activation=args.activation,
        lr=args.lr,
        batch_size=args.batch_size,
        milestones=tuple(args.milestones),
        gamma=args.gamma,
        loss_weights={"pde": args.w_pde, "ic": args.w_ic, "bc": args.w_bc},
        log_every=args.log_every,
        seed=args.seed,
        device=device,
    )
    print(f"device: {device_description(device)}", flush=True)

    def _log(epoch, m):
        print(
            f"[{epoch:6d}] pde={m.loss_pde_history[-1]:.3e} "
            f"ic={m.loss_ic_history[-1]:.3e} "
            f"bc={m.loss_bc_history[-1]:.3e}",
            flush=True,
        )

    t0 = time.time()
    model, data, config = train_pinn(
        problem, config, verbose=False, callback=None if args.quiet else _log
    )
    train_seconds = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())

    x = np.linspace(problem.x_min, problem.x_max, args.nx)
    t = np.linspace(problem.t_min, problem.t_max, args.nt)
    U = predict_grid(model, problem, x, t)

    ref, ref_err = reference.richardson_reference(
        kind=args.kind,
        nu=args.nu,
        dx=args.ref_dx,
        t_end=args.t_end,
        p=2.0,
        ic_fn=numpy_ic(args.ic_eps),
        boundary_fn=numpy_boundary(problem.bc_omegas),
    )
    ref.error_estimate = ref_err
    T, X = np.meshgrid(t, x, indexing="ij")
    U_ref = ref.u(X, T)

    err = U - U_ref
    metrics = {
        "l1": float(np.mean(np.abs(err))),
        "l2": float(np.sqrt(np.mean(err**2))),
        "linf": float(np.max(np.abs(err))),
        "ref_error_estimate": ref.error_estimate,
    }
    print(
        f"PINN vs reference: L1={metrics['l1']:.3e} "
        f"L2={metrics['l2']:.3e} Linf={metrics['linf']:.3e}"
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out + ".npz",
        x=x,
        t=t,
        U=U,
        U_ref=U_ref,
        loss_pde=np.array(model.loss_pde_history),
        loss_ic=np.array(model.loss_ic_history),
        loss_bc=np.array(model.loss_bc_history),
        kind=args.kind,
        nu=args.nu,
        ic_eps=args.ic_eps,
        omegas=np.array(args.omegas),
        epochs=args.epochs,
        n_params=n_params,
        train_seconds=train_seconds,
        device=device_description(device),
        **{k: np.array(v) for k, v in metrics.items()},
    )
    print(f"train_seconds={train_seconds:.1f}, n_params={n_params}")
    _plot(x, t, U, U_ref, args.out)
    print(f"Saved {args.out}.npz and {args.out}.png")


def _plot(x, t, U, U_ref, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, (field, title) in zip(
        ax,
        [
            (U_ref, "Richardson-extrapolated\nLax-Wendroff"),
            (U, "PINN"),
            (np.abs(U - U_ref), "|error|"),
        ],
    ):
        im = a.pcolormesh(t, x, field.T, shading="auto")
        a.set_xlabel("$t$")
        a.set_ylabel("$x$")
        a.set_title(title)
        fig.colorbar(im, ax=a)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

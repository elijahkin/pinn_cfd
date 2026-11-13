"""Experiment driver: PINN vs classical finite-volume methods vs reference.

Builds (and caches) a reference for each case, evaluates the classical methods
at several grids, and evaluates any available PINN run. Writes a tidy CSV and
comparison figures.

Examples
--------
List classical results for the full matrix::

    python -m pinn_cfd.experiments.run_matrix --classical-only

Include PINN runs found under --pinn-dir (named <kind>_nu<nu>.npz)::

    python -m pinn_cfd.experiments.run_matrix
"""

import argparse
import csv
import os

import numpy as np

from pinn_cfd import metrics
from pinn_cfd.classic import convdiff1d, fv1d, reference

PI = np.pi
DEFAULT_CASES = [
    ("linear", 0.012),
    ("linear", 0.004),
    ("linear", 0.002),
    ("nonlinear", 0.012),
    ("nonlinear", 0.004),
    ("nonlinear", 0.002),
]


def numpy_boundary(omegas=(4.0, 8.0)):
    def bc(t):
        out = np.ones_like(np.asarray(t, dtype=float))
        for w in omegas:
            out = out - 0.25 * np.cos(w * t)
        return out

    return bc


def numpy_step_ic(eps=0.0):
    if eps <= 0.0:
        return lambda x: np.where(np.asarray(x) < 0.5 * PI, 0.5, 1.5)
    return lambda x: 1.0 + 0.5 * np.tanh((np.asarray(x) - 0.5 * PI) / eps)


def parse_cases(spec):
    if spec is None:
        return DEFAULT_CASES
    cases = []
    for item in spec:
        kind, nu = item.split(":")
        cases.append((kind, float(nu)))
    return cases


def get_reference(kind, nu, t_end, ref_dx, cache_dir, boundary_fn, ic_fn):
    """Return the high-accuracy reference for a case.

    Prefers the three-grid Richardson reference (``ref3_*``), which cancels the
    leading first-order front error of the limited scheme and is accurate to
    ~1e-5; falls back to building it, then to the older two-grid ``p=2``
    reference.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path3 = os.path.join(cache_dir, f"ref3_{kind}_nu{nu:g}.npz")
    if os.path.exists(path3):
        return reference.ReferenceSolution.load(path3)
    path = os.path.join(cache_dir, f"ref_{kind}_nu{nu:g}.npz")
    if os.path.exists(path):
        return reference.ReferenceSolution.load(path)
    ref, err = reference.richardson_reference3(
        kind=kind,
        nu=nu,
        t_end=t_end,
        dx=PI / 1600,
        boundary_fn=boundary_fn,
        ic_fn=ic_fn,
    )
    ref.error_estimate = err
    ref.save(path3)
    return ref


def classical_field(kind, nu, dx, t_end, dtdx, method, boundary_fn, ic_fn):
    nsteps = int(round(t_end / (dtdx * dx)))
    if nu == 0.0:
        return fv1d.solve(
            kind=kind,
            method=method,
            dx=dx,
            nsteps=nsteps,
            dtdx=dtdx,
            boundary_fn=boundary_fn,
        )
    return convdiff1d.solve(
        kind=kind,
        method=method,
        nu=nu,
        dx=dx,
        nsteps=nsteps,
        dtdx=dtdx,
        boundary_fn=boundary_fn,
        ic_fn=ic_fn,
    )


def classical_spacetime(kind, nu, dx, t_end, dtdx, method, boundary_fn, ic_fn):
    """Whole space-time field of the convection-diffusion solve.

    Returns ``(x, t, U)`` with ``U[i, j] = u(x_j, t_i)`` over the full time
    interval, so the error can be scored over the whole grid (the metric used
    in the paper) rather than only at the final time.
    """
    nsteps = int(round(t_end / (dtdx * dx)))
    steps = list(range(1, nsteps + 1))
    res = convdiff1d.solve(
        kind=kind,
        method=method,
        nu=nu,
        dx=dx,
        nsteps=nsteps,
        dtdx=dtdx,
        boundary_fn=boundary_fn,
        ic_fn=ic_fn,
        snapshots=steps,
    )
    t = np.array([0.0] + [s * res["dt"] for s in steps])
    U = np.vstack([res["u_init"]] + [res["snapshots"][s] for s in steps])
    return res["x"], t, U


def pinn_field(npz_path):
    d = np.load(npz_path)
    return d["x"], d["t"], d["U"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="subset like 'linear:0.012 nonlinear:0.004'",
    )
    p.add_argument("--t-end", type=float, default=PI)
    p.add_argument("--dtdx", type=float, default=0.5)
    p.add_argument(
        "--grids", nargs="+", type=float, default=[PI / 50, PI / 100, PI / 200]
    )
    p.add_argument("--ref-dx", type=float, default=PI / 800)
    p.add_argument("--methods", nargs="+", type=int, default=[1, 2])
    p.add_argument("--cache-dir", default="pinn_cfd/results/cache")
    p.add_argument("--pinn-dir", default="pinn_cfd/results/pinn_runs")
    p.add_argument("--out", default="pinn_cfd/results/matrix.csv")
    p.add_argument("--classical-only", action="store_true")
    p.add_argument("--fig-dir", default="pinn_cfd/results/figures")
    args = p.parse_args()

    cases = parse_cases(args.cases)
    boundary_fn = numpy_boundary()
    ic_fn = numpy_step_ic()
    rows = []

    for kind, nu in cases:
        ref = get_reference(
            kind, nu, args.t_end, args.ref_dx, args.cache_dir, boundary_fn, ic_fn
        )
        print(
            f"\n=== {kind}, nu={nu:g} (ref error est. "
            f"{ref.error_estimate if ref.error_estimate else float('nan'):.2e}) ==="
        )

        for method in args.methods:
            for dx in args.grids:
                xs, ts, Us = classical_spacetime(
                    kind, nu, dx, args.t_end, args.dtdx, method, boundary_fn, ic_fn
                )
                Ts, Xs = np.meshgrid(ts, xs, indexing="ij")
                m = metrics.summarize(Us, ref.u(Xs, Ts))
                m_final = metrics.summarize(Us[-1], ref.u(xs, ts[-1]))
                rows.append(
                    {
                        "case": f"{kind}_nu{nu:g}",
                        "kind": kind,
                        "nu": nu,
                        "solver": "classical",
                        "method": method,
                        "dx": dx,
                        "nsteps": len(ts) - 1,
                        **m,
                        "l1_final": m_final["l1"],
                        "l2_final": m_final["l2"],
                        "linf_final": m_final["linf"],
                    }
                )
                solver_name = {1: "Lax-Wendroff", 2: "3-point Central"}.get(
                    method, f"method {method}"
                )
                print(
                    f"  classical {solver_name} dx=pi/{PI/dx:.0f}: "
                    f"grid L1={m['l1']:.3e} Linf={m['linf']:.3e} | "
                    f"final L1={m_final['l1']:.3e} Linf={m_final['linf']:.3e}"
                )

        if not args.classical_only:
            npz = os.path.join(args.pinn_dir, f"{kind}_nu{nu:g}.npz")
            if os.path.exists(npz):
                x, t, U = pinn_field(npz)
                T, X = np.meshgrid(t, x, indexing="ij")
                Uref = ref.u(X, T)
                m = metrics.summarize(U, Uref)
                m_final = metrics.summarize(U[-1], Uref[-1])
                rows.append(
                    {
                        "case": f"{kind}_nu{nu:g}",
                        "kind": kind,
                        "nu": nu,
                        "solver": "pinn",
                        "method": 0,
                        "dx": float("nan"),
                        "nsteps": len(t),
                        **m,
                        "l1_final": m_final["l1"],
                        "l2_final": m_final["l2"],
                        "linf_final": m_final["linf"],
                    }
                )
                print(
                    f"  PINN: grid L1={m['l1']:.3e} Linf={m['linf']:.3e} | "
                    f"final L1={m_final['l1']:.3e} Linf={m_final['linf']:.3e}"
                )
                _save_case_figure(
                    kind, nu, args.t_end, ref, x, t, U, Uref, args.fig_dir
                )
            else:
                print(f"  PINN: no run at {npz}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    keys = [
        "case",
        "kind",
        "nu",
        "solver",
        "method",
        "dx",
        "nsteps",
        "l1",
        "l2",
        "linf",
        "rel_l1",
        "l1_final",
        "l2_final",
        "linf_final",
    ]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {args.out} ({len(rows)} rows)")


def _save_case_figure(kind, nu, t_end, ref, x, t, U, Uref, fig_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, (field, title) in zip(
        ax,
        [
            (Uref, "Richardson-extrapolated\nLax-Wendroff"),
            (U, "PINN"),
            (np.abs(U - Uref), "|error|"),
        ],
    ):
        im = a.pcolormesh(t, x, field.T, shading="auto")
        a.set_xlabel("$t$")
        a.set_ylabel("$x$")
        a.set_title(title)
        fig.colorbar(im, ax=a)
    fig.tight_layout()
    path = os.path.join(fig_dir, f"{kind}_nu{nu:g}_pinn.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

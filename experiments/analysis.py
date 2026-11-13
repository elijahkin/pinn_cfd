"""Phase 5 analysis: convergence orders, cost, spectral bias, nu->0 stress test.

Reads the artifacts produced by run_matrix.py and run_pinn.py and prints a
text report, plus a couple of summary figures.

    python -m pinn_cfd.experiments.analysis
"""

import csv
import os
import time

import numpy as np

from pinn_cfd import metrics
from pinn_cfd.classic import convdiff1d
from pinn_cfd.experiments.run_matrix import (
    classical_spacetime,
    get_reference,
    numpy_boundary,
    numpy_step_ic,
)

PI = np.pi
RUNS = "pinn_cfd/results/pinn_runs"
CACHE = "pinn_cfd/results/cache"
FIGS = "pinn_cfd/results/figures"


def load_metrics(path):
    if not os.path.exists(path):
        return None
    d = np.load(path)
    out = {k: float(d[k]) for k in ("l1", "l2", "linf") if k in d}
    for k in ("epochs", "n_params", "train_seconds"):
        if k in d:
            out[k] = float(d[k])
    return out


def convergence_orders():
    print("== Convergence orders (final-time L1 vs dx, from matrix.csv) ==")
    rows = list(csv.DictReader(open("pinn_cfd/results/matrix.csv")))
    cases = sorted({r["case"] for r in rows})
    for case in cases:
        parts = []
        for method in ("1", "2"):
            sub = [
                r
                for r in rows
                if r["case"] == case
                and r["solver"] == "classical"
                and r["method"] == method
            ]
            if len(sub) < 2:
                continue
            sub.sort(key=lambda r: float(r["dx"]))
            dx = np.array([float(r["dx"]) for r in sub])
            l1 = np.array([float(r["l1_final"]) for r in sub])
            if np.any(l1 <= 0):
                continue
            slope = np.polyfit(np.log(dx), np.log(l1), 1)[0]
            parts.append(f"m{method}={slope:.2f}")
        print(f"  {case:<20} " + "  ".join(parts))


def cost_table():
    print("\n== Cost ==")
    t0 = time.time()
    nsteps = int(round(PI / (0.5 * (PI / 200))))
    convdiff1d.solve(kind="linear", method=1, nu=0.012, dx=PI / 200, nsteps=nsteps)
    dt_classical = time.time() - t0
    print(
        f"  classical Lax-Wendroff (dx=pi/200, N={nsteps}): {dt_classical*1e3:.1f} ms"
    )

    for name in (
        "linear_nu0.012_smooth_om4",
        "linear_nu0.012_smooth_om412",
        "linear_nu0.004",
        "nonlinear_nu0.012",
    ):
        m = load_metrics(os.path.join(RUNS, name + ".npz"))
        if m and "train_seconds" in m:
            print(
                f"  PINN {name:<18}: {m['n_params']:.0f} params, "
                f"{m['epochs']:.0f} epochs, {m['train_seconds']/60:.1f} min"
            )
        elif m:
            print(
                f"  PINN {name:<18}: metrics available "
                f"(L1={m['l1']:.2e}); timing not recorded"
            )


def spectral_bias():
    print("\n== Spectral bias: Dirichlet forcing frequencies ==")
    # The smooth-IC family isolates the effect of the forcing frequency; the
    # jump-IC family is dominated by the discontinuity (Linf ~ 0.9).
    cases = [
        ("smooth\n$\\omega=4$", "linear_nu0.012_smooth_om4"),
        ("smooth\n$\\omega=4,8$", "linear_nu0.012_smooth"),
        ("smooth\n$\\omega=4,12$", "linear_nu0.012_smooth_om412"),
        ("jump\n$\\omega=4$", "linear_nu0.012_om4"),
        ("jump\n$\\omega=4,8$", "linear_nu0.012"),
        ("jump\n$\\omega=4,12$", "linear_nu0.012_om412"),
    ]
    results = []
    for label, name in cases:
        l1s = []
        for suffix in ("", "_seed1", "_seed2"):
            m = load_metrics(os.path.join(RUNS, name + suffix + ".npz"))
            if m:
                l1s.append(m["l1"])
        if l1s:
            l1s = np.array(l1s)
            results.append((label, float(l1s.mean()), float(l1s.std(ddof=0))))
            flat = label.replace("\n", ", ").replace("$", "").replace("\\omega", "w")
            print(
                f"  {flat:<14} L1={l1s.mean():.3e}+-{l1s.std(ddof=0):.1e} (n={len(l1s)})"
            )
    if results:
        _bar(
            [r[0] for r in results],
            [r[1] for r in results],
            [r[2] for r in results],
            "PINN L1 error vs BC frequency (mean of seeds)",
            os.path.join(FIGS, "analysis_spectral_bias.png"),
        )
    return results


def stress_test():
    print("\n== nu -> 0 stress test (nu = 1e-4) ==")
    for kind in ("linear", "nonlinear"):
        m = load_metrics(os.path.join(RUNS, f"{kind}_nu0.0001.npz"))
        if not m:
            print(f"  {kind}: no PINN run yet")
            continue
        ref = get_reference(
            kind, 1e-4, PI, PI / 800, CACHE, numpy_boundary(), numpy_step_ic()
        )
        dx = PI / 200
        nsteps = int(round(PI / (0.5 * dx)))
        xs, ts, Us = classical_spacetime(
            kind, 1e-4, dx, PI, 0.5, 1, numpy_boundary(), numpy_step_ic()
        )
        Ts, Xs = np.meshgrid(ts, xs, indexing="ij")
        cm = metrics.summarize(Us, ref.u(Xs, Ts))
        print(
            f"  {kind}: PINN L1={m['l1']:.3e} Linf={m['linf']:.3e} | "
            f"classical Lax-Wendroff (pi/200) "
            f"grid L1={cm['l1']:.3e} Linf={cm['linf']:.3e}"
        )

        # Snapshot figure.
        d = np.load(os.path.join(RUNS, f"{kind}_nu0.0001.npz"))
        res = {"x": xs, "u_final": Us[-1]}
        _snapshot(
            kind, 1e-4, ref, d, res, os.path.join(FIGS, f"analysis_stress_{kind}.png")
        )


def _bar(labels, values, yerr, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(labels, values, yerr=yerr, capsize=3)
    ax.set_ylabel("$L^1$ error")
    ax.set_xlabel("inflow frequencies")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _snapshot(kind, nu, ref, d, classical, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    xp, tp, Up = d["x"], d["t"], d["U"]
    x = xp
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(
        x,
        ref.u(x, PI),
        "k",
        lw=2,
        label="Richardson-extrapolated\nLax-Wendroff",
    )
    ax.plot(xp, Up[-1], "r--", lw=1.5, label="PINN")
    ax.plot(
        classical["x"],
        classical["u_final"],
        "b:",
        lw=1.5,
        label="Lax-Wendroff ($\\pi/200$)",
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u$")
    ax.set_title(f"{kind}, $\\nu=10^{{-4}}$, $t=\\pi$")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    convergence_orders()
    cost_table()
    spectral_bias()
    stress_test()


if __name__ == "__main__":
    main()

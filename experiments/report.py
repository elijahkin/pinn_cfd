"""Build summary figures from the experiment matrix.

Reads results/matrix.csv (produced by run_matrix.py) and, where available,
PINN fields under results/pinn_runs/. Produces:

  * convergence_<case>.png : L1 error vs dx for each classical method, with
    the PINN L1 marked as a horizontal reference.
  * snapshot_<case>.png    : u(x) at t_end for reference, PINN, and the best
    classical method.

Usage:
    python -m pinn_cfd.experiments.report
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from pinn_cfd.classic import convdiff1d, fv1d, reference

PI = np.pi

# Human-readable names for the classical solvers (see convdiff1d.solve).
METHOD_LABELS = {1: "Lax-Wendroff", 2: "3-point Central"}


def load_matrix(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["nu"] = float(r["nu"])
            r["dx"] = float(r["dx"])
            r["method"] = int(r["method"])
            r["l1"] = float(r["l1"])
            r["linf"] = float(r["linf"])
            if "l1_final" in r and r["l1_final"] not in ("", "nan"):
                r["l1_final"] = float(r["l1_final"])
            rows.append(r)
    return rows


def numpy_boundary(omegas=(4.0, 8.0)):
    def bc(t):
        out = np.ones_like(np.asarray(t, dtype=float))
        for w in omegas:
            out = out - 0.25 * np.cos(w * t)
        return out

    return bc


def numpy_step_ic(eps=0.0):
    return lambda x: np.where(np.asarray(x) < 0.5 * PI, 0.5, 1.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="pinn_cfd/results/matrix.csv")
    p.add_argument("--pinn-dir", default="pinn_cfd/results/pinn_runs")
    p.add_argument("--cache-dir", default="pinn_cfd/results/cache")
    p.add_argument("--fig-dir", default="pinn_cfd/results/figures")
    p.add_argument("--t-end", type=float, default=PI)
    p.add_argument("--dtdx", type=float, default=0.5)
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_matrix(args.matrix)
    os.makedirs(args.fig_dir, exist_ok=True)
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case"]].append(r)

    for case, rs in sorted(by_case.items()):
        kind = rs[0]["kind"]
        nu = rs[0]["nu"]

        # --- convergence plot -------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 3.6))
        classical = [r for r in rs if r["solver"] == "classical"]
        for method in sorted({r["method"] for r in classical}):
            sub = sorted(
                [r for r in classical if r["method"] == method], key=lambda r: r["dx"]
            )
            if len(sub) < 2:
                continue
            dx = np.array([r["dx"] for r in sub])
            l1 = np.array([r["l1_final"] for r in sub])
            ax.loglog(dx, l1, "o-", label=METHOD_LABELS.get(method, f"method {method}"))

        pinn = [r for r in rs if r["solver"] == "pinn"]
        if pinn:
            ax.axhline(pinn[0]["l1_final"], color="k", ls="--", label="PINN")

        ax.set_xlabel(r"$\Delta x$")
        ax.set_ylabel(r"final-time $L^1$ error")
        ax.set_title(f"{kind}, $\\nu={nu:g}$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            os.path.join(args.fig_dir, f"convergence_{case}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        # --- snapshot plot ----------------------------------------------------
        npz = os.path.join(args.pinn_dir, f"{kind}_nu{nu:g}.npz")
        ref_path = os.path.join(args.cache_dir, f"ref3_{kind}_nu{nu:g}.npz")
        if not os.path.exists(ref_path):
            ref_path = os.path.join(args.cache_dir, f"ref_{kind}_nu{nu:g}.npz")
        if not (os.path.exists(npz) and os.path.exists(ref_path)):
            continue
        ref = reference.ReferenceSolution.load(ref_path)
        d = np.load(npz)
        xp, tp, Up = d["x"], d["t"], d["U"]
        x = xp
        uref = ref.u(x, args.t_end)
        upinn = Up[-1]
        dx_c = PI / 200
        res = (
            convdiff1d.solve(
                kind=kind,
                method=1,
                nu=nu,
                dx=dx_c,
                nsteps=int(round(args.t_end / (args.dtdx * dx_c))),
                boundary_fn=numpy_boundary(),
                ic_fn=numpy_step_ic(),
            )
            if nu > 0
            else fv1d.solve(
                kind=kind,
                method=3,
                dx=dx_c,
                nsteps=int(round(args.t_end / (args.dtdx * dx_c))),
            )
        )

        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(
            x,
            uref,
            "k",
            lw=2,
            label="Richardson-extrapolated\nLax-Wendroff",
        )
        ax.plot(xp, upinn, "r--", lw=1.5, label="PINN")
        ax.plot(
            res["x"],
            res["u_final"],
            "b:",
            lw=1.5,
            label="Lax-Wendroff ($\\pi/200$)",
        )
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")
        ax.set_title(f"{kind}, $\\nu={nu:g}$, $t=\\pi$")
        ax.legend(fontsize=8.5)
        fig.tight_layout()
        fig.savefig(
            os.path.join(args.fig_dir, f"snapshot_{case}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
        print(f"wrote figures for {case}")


if __name__ == "__main__":
    main()

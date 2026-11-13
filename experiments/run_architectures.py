"""PINN architecture comparison.

Trains several PINN variants on the same problem and training budget, scores
each against the Richardson reference, and writes a tidy CSV plus figures.

Plain MLP variants use the same tanh activation and vary only width and depth.
Four alternatives are tested at 4x64: a sine activation (SIREN), a gated
("modified") MLP, and two random Fourier-feature embeddings. Each run records
its L1 error against the reference every ``--eval-every`` epochs, which is
used for the convergence-over-time figure.

Usage
-----
Train one variant::

    python -m pinn_cfd.experiments.run_architectures train \
        --case linear --run gated_4x64

Summarize everything found under --out-dir::

    python -m pinn_cfd.experiments.run_architectures summarize --case linear \
        --figures
"""

import argparse
import csv
import os
import time

import numpy as np

from pinn_cfd import metrics
from pinn_cfd.pinn import (
    ConvectionDiffusionProblem,
    TrainConfig,
    device_description,
    predict_grid,
    resolve_device,
    train_pinn,
)
from pinn_cfd.experiments.run_matrix import get_reference, numpy_boundary, numpy_step_ic

PI = np.pi
CACHE = "pinn_cfd/results/cache"
OUT_DIR = "pinn_cfd/results/arch_runs"
FIGS = "pinn_cfd/results/figures"

# Baseline recipe, matching the paper's 3000-epoch GPU budget.
BASE = dict(
    num_epochs=3000,
    N_PDE=3000,
    N_IC=128,
    N_BC=128,
    hidden_size=64,
    num_hidden_layers=4,
    activation="tanh",
    hard_ic=False,
    lr=1e-3,
    batch_size=256,
    milestones=(1200, 2200),
    gamma=0.5,
    loss_weights={"pde": 1.0, "ic": 10.0, "bc": 1.0},
    weighting="fixed",
    log_every=50,
    seed=0,
)

# name -> (group, config overrides). Capacity spans ~3k-66k parameters.
VARIANTS = {
    "depth2_width32": ("capacity", {"num_hidden_layers": 2, "hidden_size": 32}),
    "depth4_width64": ("capacity", {}),
    "depth6_width64": ("capacity", {"num_hidden_layers": 6, "hidden_size": 64}),
    "depth4_width128": ("capacity", {"hidden_size": 128}),
    "sine_2x32": (
        "capacity",
        {"activation": "sin", "num_hidden_layers": 2, "hidden_size": 32, "siren_omega": 3.0},
    ),
    "sine_4x64": ("capacity", {"activation": "sin", "siren_omega": 3.0}),
    "sine_6x64": (
        "capacity",
        {"activation": "sin", "num_hidden_layers": 6, "hidden_size": 64, "siren_omega": 3.0},
    ),
    "sine_4x128": (
        "capacity",
        {"activation": "sin", "num_hidden_layers": 4, "hidden_size": 128, "siren_omega": 3.0},
    ),
    "gated_4x64": ("gated", {"arch": "modified"}),
    "fourier_m64": ("fourier", {"fourier_features": 64, "fourier_sigma": 1.0}),
    "fourier_m128": ("fourier", {"fourier_features": 128, "fourier_sigma": 1.0}),
}

# Seeds trained for every variant. Seed 0 keeps the legacy filename for
# backward compatibility; other seeds are suffixed.
SEEDS = (0, 1, 2, 3, 4)

GROUP_ORDER = ("capacity", "activation", "gated", "fourier")
COLORS = {
    "depth2_width32": "tab:blue",
    "depth4_width64": "tab:orange",
    "depth6_width64": "tab:green",
    "depth4_width128": "tab:red",
    "sine_2x32": "tab:cyan",
    "sine_4x64": "tab:purple",
    "sine_6x64": "tab:olive",
    "sine_4x128": "tab:gray",
    "gated_4x64": "black",
    "fourier_m64": "tab:brown",
    "fourier_m128": "tab:pink",
}


def build_config(overrides):
    cfg = dict(BASE)
    cfg.update(overrides)
    cfg["milestones"] = tuple(cfg["milestones"])
    cfg["loss_weights"] = dict(cfg["loss_weights"])
    return TrainConfig(**cfg)


def run_path(kind, name, out_dir=OUT_DIR, nu=0.012, seed=0):
    suffix = "" if seed == 0 else f"_seed{seed}"
    return os.path.join(out_dir, f"{kind}_nu{nu:g}_{name}{suffix}.npz")


def train_one(
    kind,
    name,
    nu=0.012,
    epochs=None,
    out_dir=OUT_DIR,
    nx=201,
    nt=101,
    device=None,
    seed=0,
    hard_ic=False,
):
    group, overrides = VARIANTS[name]
    overrides = dict(overrides)
    problem = ConvectionDiffusionProblem(kind=kind, nu=nu)
    overrides["seed"] = seed
    overrides["hard_ic"] = bool(hard_ic)
    config = build_config(overrides)
    config.device = resolve_device(device)
    if epochs is not None:
        config.num_epochs = epochs
    print(f"{kind}/{name}: device={device_description(config.device)}", flush=True)

    x = np.linspace(problem.x_min, problem.x_max, nx)
    t = np.linspace(problem.t_min, problem.t_max, nt)
    ref = get_reference(
        kind, nu, PI, PI / 800, CACHE, numpy_boundary(), numpy_step_ic()
    )
    T, X = np.meshgrid(t, x, indexing="ij")
    U_ref = ref.u(X, T)

    err_history = []

    def record(epoch, model):
        U = predict_grid(model, problem, x, t)
        err_history.append((epoch, metrics.l1(U, U_ref), metrics.linf(U, U_ref)))

    t0 = time.time()
    model, data, config = train_pinn(problem, config, verbose=False, callback=record)
    train_seconds = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())

    U = predict_grid(model, problem, x, t)
    m = metrics.summarize(U, U_ref)

    os.makedirs(out_dir, exist_ok=True)
    path = run_path(kind, name, out_dir, nu, seed)
    np.savez_compressed(
        path,
        x=x,
        t=t,
        U=U,
        U_ref=U_ref,
        loss_pde=np.array(model.loss_pde_history),
        loss_ic=np.array(model.loss_ic_history),
        loss_bc=np.array(model.loss_bc_history),
        err_history=np.array(err_history),
        group=group,
        run=name,
        kind=kind,
        nu=nu,
        seed=seed,
        activation=config.activation,
        arch=config.arch,
        hard_ic=config.hard_ic,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        epochs=config.num_epochs,
        n_params=n_params,
        train_seconds=train_seconds,
        device=device_description(config.device),
        **{k: np.array(v) for k, v in m.items()},
    )
    print(
        f"{kind}/{name:<16} seed={seed} params={n_params:>6d} "
        f"L1={m['l1']:.3e} L2={m['l2']:.3e} Linf={m['linf']:.3e} "
        f"({train_seconds/60:.1f} min)",
        flush=True,
    )
    return m


def load_runs(kind, out_dir=OUT_DIR, nu=0.012, seeds=SEEDS):
    rows = []
    for name in VARIANTS:
        for seed in seeds:
            path = run_path(kind, name, out_dir, nu, seed)
            if not os.path.exists(path):
                continue
            d = np.load(path)
            rows.append(
                {
                    "case": f"{kind}_nu{nu:g}",
                    "group": str(d["group"]),
                    "run": name,
                    "seed": int(d["seed"]) if "seed" in d else 0,
                    "activation": str(d["activation"]),
                    "arch": str(d["arch"]) if "arch" in d else "mlp",
                    "hard_ic": bool(d["hard_ic"]) if "hard_ic" in d else False,
                    "layers": int(d["num_hidden_layers"]),
                    "hidden": int(d["hidden_size"]),
                    "n_params": int(d["n_params"]),
                    "epochs": int(d["epochs"]),
                    "train_seconds": float(d["train_seconds"]),
                    "l1": float(d["l1"]),
                    "l2": float(d["l2"]),
                    "linf": float(d["linf"]),
                }
            )
    return rows


def aggregate(rows):
    """Collapse per-seed rows into one row per (case, run) with mean and std."""
    by_run = {}
    for r in rows:
        by_run.setdefault((r["case"], r["run"]), []).append(r)
    agg = []
    for (case, name), rs in by_run.items():
        l1 = np.array([r["l1"] for r in rs])
        linf = np.array([r["linf"] for r in rs])
        base = dict(rs[0])
        base.update(
            n_seeds=len(rs),
            l1=float(l1.mean()),
            l1_std=float(l1.std(ddof=0)),
            l1_min=float(l1.min()),
            l1_max=float(l1.max()),
            linf=float(linf.mean()),
            linf_std=float(linf.std(ddof=0)),
            linf_min=float(linf.min()),
            linf_max=float(linf.max()),
        )
        agg.append(base)
    return agg


def load_error_history(kind, name, out_dir=OUT_DIR, seed=0):
    path = run_path(kind, name, out_dir, seed=seed)
    if not os.path.exists(path):
        return None
    d = np.load(path)
    if "err_history" not in d or d["err_history"].size == 0:
        return None
    return d["err_history"]


def write_csv(kind, rows, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"architectures_{kind}.csv")
    keys = [
        "case",
        "group",
        "run",
        "activation",
        "arch",
        "hard_ic",
        "layers",
        "hidden",
        "n_params",
        "epochs",
        "n_seeds",
        "l1",
        "l1_std",
        "l1_min",
        "l1_max",
        "linf",
        "linf_std",
        "linf_min",
        "linf_max",
        "train_seconds",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def write_seed_csv(kind, rows, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"architectures_{kind}_seeds.csv")
    keys = [
        "case",
        "run",
        "seed",
        "n_params",
        "epochs",
        "train_seconds",
        "l1",
        "l2",
        "linf",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def summarize(kind, out_dir=OUT_DIR):
    per_seed = load_runs(kind, out_dir)
    if not per_seed:
        print(f"no runs found for {kind}")
        return per_seed
    rows = aggregate(per_seed)
    rows.sort(key=lambda r: r["n_params"])
    print(f"\n== Architecture comparison: {kind} (mean over seeds) ==")
    for r in rows:
        arch = f"{r['layers']}x{r['hidden']}"
        if r["arch"] == "modified":
            arch = "gated " + arch
        print(
            f"    {r['run']:<16} {arch:<12} params={r['n_params']:>6d} "
            f"n={r['n_seeds']} L1={r['l1']:.3e}+-{r['l1_std']:.1e} "
            f"Linf={r['linf']:.3e}+-{r['linf_std']:.1e}"
        )
    path = write_csv(kind, rows, out_dir)
    seed_path = write_seed_csv(kind, per_seed, out_dir)
    print(f"\n  wrote {path}")
    print(f"  wrote {seed_path}")
    return rows


def _figures(kind, out_dir=OUT_DIR):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = {r["run"]: r for r in aggregate(load_runs(kind, out_dir))}
    if not rows:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    names = sorted(rows, key=lambda n: rows[n]["n_params"])
    ax1.bar(
        names,
        [rows[n]["l1"] for n in names],
        yerr=[rows[n]["l1_std"] for n in names],
        capsize=3,
        color=[COLORS[n] for n in names],
    )
    ax1.set_ylabel(r"$L^1$ error (whole grid)")
    ax1.set_title("mean over seeds")
    ax1.tick_params(axis="x", rotation=30)

    for name in names:
        h = load_error_history(kind, name, out_dir, seed=0)
        if h is None:
            continue
        ax2.semilogy(h[:, 0], h[:, 1], "-", color=COLORS[name], label=name)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel(r"$L^1$ error")
    ax2.set_title("convergence (seed 0)")
    ax2.legend(fontsize=8)

    fig.suptitle(f"PINN architectures ({kind}, $\\nu=0.012$)")
    fig.tight_layout()
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, f"architecture_{kind}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _compare_figure(cases=("linear", "nonlinear"), out_dir=OUT_DIR):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loaded = {k: {r["run"]: r for r in aggregate(load_runs(k, out_dir))} for k in cases}
    if not any(loaded.values()):
        return
    names = sorted(
        {n for d in loaded.values() for n in d},
        key=lambda n: loaded["linear"].get(n, loaded["nonlinear"].get(n))["n_params"],
    )

    fig, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 4), sharey=True)
    if len(cases) == 1:
        axes = [axes]
    for ax, kind in zip(axes, cases):
        rows = loaded[kind]
        present = [n for n in names if n in rows]
        ax.bar(
            present,
            [rows[n]["l1"] for n in present],
            yerr=[rows[n]["l1_std"] for n in present],
            capsize=3,
            color=[COLORS[n] for n in present],
        )
        ax.set_title(kind)
        ax.set_ylabel(r"$L^1$ error (whole grid)")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, "architecture_compare.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _convergence_figure(cases=("linear", "nonlinear"), out_dir=OUT_DIR):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 4), sharey=True)
    if len(cases) == 1:
        axes = [axes]
    for ax, kind in zip(axes, cases):
        for name in VARIANTS:
            curves = []
            for seed in SEEDS:
                h = load_error_history(kind, name, out_dir, seed=seed)
                if h is None or h.size == 0:
                    continue
                curves.append(h)
            if not curves:
                continue
            n = min(c.shape[0] for c in curves)
            epochs = curves[0][:n, 0]
            vals = np.vstack([c[:n, 1] for c in curves])
            mean = vals.mean(axis=0)
            std = vals.std(axis=0)
            ax.semilogy(epochs, mean, "-", color=COLORS[name], label=name)
            ax.fill_between(
                epochs,
                mean - std,
                mean + std,
                color=COLORS[name],
                alpha=0.15,
                linewidth=0,
            )
        ax.set_xlabel("epoch")
        ax.set_title(kind)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$L^1$ error")
    fig.suptitle("Mean over five seeds (shaded: $\\pm1$ s.d.)")
    fig.tight_layout()
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, "architecture_convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--case", default="linear", choices=["linear", "nonlinear"])
    pt.add_argument("--run", required=True, choices=sorted(VARIANTS))
    pt.add_argument("--nu", type=float, default=0.012)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--epochs", type=int, default=None)
    pt.add_argument("--hard-ic", action="store_true", default=False)
    pt.add_argument("--device", default=None, help="torch device, e.g. 'cuda'")
    pt.add_argument("--out-dir", default=OUT_DIR)

    ps = sub.add_parser("summarize")
    ps.add_argument("--case", default="linear", choices=["linear", "nonlinear"])
    ps.add_argument("--out-dir", default=OUT_DIR)
    ps.add_argument("--figures", action="store_true")

    args = p.parse_args()
    if args.cmd == "train":
        train_one(
            args.case,
            args.run,
            nu=args.nu,
            epochs=args.epochs,
            out_dir=args.out_dir,
            device=args.device,
            seed=args.seed,
            hard_ic=args.hard_ic,
        )
    else:
        summarize(args.case, args.out_dir)
        if args.figures:
            _figures(args.case, args.out_dir)
            _compare_figure(out_dir=args.out_dir)
            _convergence_figure(out_dir=args.out_dir)


if __name__ == "__main__":
    main()

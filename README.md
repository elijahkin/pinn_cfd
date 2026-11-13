# pinn_cfd

Physics-informed neural networks (PINNs) for the one-dimensional
convection--diffusion equation, compared against classical finite-volume methods
on problems with sharp features (a discontinuous initial condition, an
oscillatory Dirichlet inflow, and a zero-gradient outflow).

The project pairs a PyTorch PINN with numpy implementations of the classical
finite-volume schemes, a high-accuracy reference solution, and scripts that
reproduce the comparison in the accompanying paper.

## Layout

```
pinn_cfd/
  classic/          # classical solvers and the reference
    fv1d.py           explicit convection methods (Euler upwind, LW, LW+MC, RK3)
    convdiff1d.py     implicit convection-diffusion solver (limited LW)
    limiters.py       minmod, van Leer MC
    reference.py      fine-grid + Richardson references, interpolation
    __init__.py
  pinn/             # the PINN
    problem.py        PDE/IC/BC definition
    model.py          MLP variants (Fourier features, sine, gated), hard
                      initial condition, adaptive loss weighting (gradnorm,
                      inverse-loss balancing)
    data.py           collocation-point sampling
    losses.py         convection-diffusion residual loss
    train.py          training driver (Adam + step LR)
  metrics.py        # L1 / L2 / L-inf / relative-L1
  experiments/      # drivers and validation
    validate_classic.py
    validate_reference.py
    validate_pinn.py
    run_pinn.py       train a single PINN and score it
    run_matrix.py     full comparison matrix -> results/matrix.csv
    report.py         convergence/snapshot figures
    analysis.py       orders, cost, frequency ablation, nu -> 0 stress test
    run_architectures.py  PINN sweep: width/depth, sine, gated MLP, Fourier
                          features -> results/arch_runs, CSV, figures
  results/          # curated summary CSVs; generated runs/cache are ignored
  paper/            # LaTeX paper (main.tex, figures/, references.bib)
```

## Requirements

Python 3.10+. Install the Python dependencies with

```bash
pip install -r requirements.txt
```

Building the paper additionally needs a LaTeX distribution (`pdflatex`,
`latexmk`, `bibtex`).

## Running

The top-level importable package is `pinn_cfd`, so commands must be run from the
directory that **contains** the `pinn_cfd/` folder (that is, the parent of the
repository checkout). For example, from `/home/eli/GitHub`:

```bash
# correctness checks
python -m pinn_cfd.experiments.validate_classic
python -m pinn_cfd.experiments.validate_reference
python -m pinn_cfd.experiments.validate_pinn

# train one PINN and score it against the reference
python -m pinn_cfd.experiments.run_pinn \
    --kind linear --nu 0.012 --epochs 3000 --N-pde 3000 \
    --hidden 64 --layers 4 --batch-size 256 --milestones 1200 2200 \
    --out pinn_cfd/results/pinn_runs/linear_nu0.012

# full comparison matrix + figures (uses any PINN runs in results/pinn_runs)
python -m pinn_cfd.experiments.run_matrix
python -m pinn_cfd.experiments.report
python -m pinn_cfd.experiments.analysis

# architecture comparison: train one variant, then summarize the sweep
python -m pinn_cfd.experiments.run_architectures train --case linear --run sine_4x64
python -m pinn_cfd.experiments.run_architectures summarize --case linear --figures
```

The architecture sweep varies width/depth of a tanh MLP (2x32, 4x64, 6x64,
4x128) and compares four modifications at 4x64: a sine activation (SIREN-style),
a gated ("modified") MLP, and random Fourier-feature embeddings (m=64, m=128),
all under the paper's 3000-epoch budget. Runs are independent, so they can be
launched in parallel, e.g.

```bash
for r in depth2_width32 depth4_width64 depth6_width64 depth4_width128 \
         sine_4x64 gated_4x64 fourier_m64 fourier_m128; do
    OMP_NUM_THREADS=1 python -m pinn_cfd.experiments.run_architectures \
        train --case linear --run "$r" &
done; wait
```

Outputs are written under `pinn_cfd/results/`.

## Paper

```bash
cd pinn_cfd/paper
latexmk -pdf main.tex
```

## What the results show

- The limited finite-volume scheme has four to nine times lower `L1` error than
  the baseline PINN at the reported resolution, at a small fraction of the cost
  (tens of ms vs. minutes).
- The reference is a three-grid Richardson extrapolation of the limited scheme
  (cancelling the first-order front error in an averaged sense), with an `L1`
  self-convergence estimate of `1.3e-5`-`5.6e-5` for `nu >= 0.002` and
  `1.4e-4`-`3.9e-4` at the stress value `nu = 1e-4`. The corresponding `Linf`
  estimate on the common evaluation grid is `5.7e-4`-`5.4e-3` and `4e-2`-`1e-1`.
- The PINN's `L-inf` error is dominated by the discontinuous initial datum.
  Imposing it exactly with a branch in the forward pass adds no parameters and
  reduces the mean `L-inf` from `0.70-0.80` (soft IC) to `0.25-0.43` (linear)
  and `0.18-0.35` (nonlinear) across eight plain architectures.
- With exact IC, the best PINN `L-inf` is `0.25-0.27` (linear) and `0.18-0.19`
  (nonlinear), still about `2.5-3` times the finite-volume values on the common
  evaluation grid. Differences among the best architectures are within one
  standard deviation.
- Exact IC does not close the `L1` or `L-inf` gap: the best exact-IC models
  remain about `2.5-3` times worse than the finite-volume scheme in both norms.
- The best architecture depends on the target: `4x64` sine for linear `L1`,
  `4x128` sine for linear `L-inf`, and `4x64` or `6x64` tanh for nonlinear
  `L-inf`. Larger or deeper sine networks do not improve the `L1` tradeoff.

## Notes

- `results/` is generated output. Only the summary CSVs used for the paper
  tables (and the stress-test logs) are tracked; raw runs, logs, figures, and
  `cache/` are ignored by git.
- `results/cache/` holds cached reference solutions; delete it to rebuild.
- PINN runs use the GPU (CUDA) when one is available and fall back to CPU
  otherwise; the experiment matrix is embarrassingly parallel across cases.
- The architecture sweep supports multiple seeds via `--seed`; saved fields can
  be re-scored against a new reference without retraining.
- The reported results were produced on an NVIDIA GeForce RTX 3060 (12 GB) with
  PyTorch 2.6.0 / CUDA 12.4; a 3000-epoch baseline run takes about 2.4 minutes.
  The hard-IC sweep was run with multiple jobs concurrently on the same GPU, so
  its per-run wall-clock times are not directly comparable.

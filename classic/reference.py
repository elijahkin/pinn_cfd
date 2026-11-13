"""High-accuracy reference solutions for the 1-D convection-diffusion problem.

The comparison methods are at most 2nd order, so we need a solution that is
an order of magnitude (or more) more accurate to score them against. Two
independent routes are provided:

  * build_reference : the verified limited finite-volume scheme run on a very
    fine grid (works for linear and nonlinear, all nu).
  * richardson_reference : Richardson-extrapolated combination of two fine
    grids, which also yields an error estimate for the reference itself.

The closed-form solution for the pure-convection (nu = 0) linear problem lives
in fv1d.exact_linear and is re-exported here.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import convdiff1d, fv1d

PI = np.pi


class ReferenceSolution:
    """A space-time field u(x,t) with bilinear interpolation."""

    def __init__(self, x, t, U, *, kind, nu, method, error_estimate=None):
        self.x = np.asarray(x, dtype=float)
        self.t = np.asarray(t, dtype=float)
        self.U = np.asarray(U, dtype=float)
        self.kind = kind
        self.nu = nu
        self.method = method
        self.error_estimate = error_estimate
        self._interp = RegularGridInterpolator(
            (self.t, self.x), self.U, bounds_error=False, fill_value=None
        )

    def u(self, xq, tq):
        """Evaluate u at query points (broadcasting xq and tq)."""
        xq, tq = np.broadcast_arrays(
            np.asarray(xq, dtype=float), np.asarray(tq, dtype=float)
        )
        pts = np.column_stack([tq.ravel(), xq.ravel()])
        out = self._interp(pts, method="linear")
        return out.reshape(xq.shape)

    def field(self, tq):
        """Return (x, u(x, tq)) on the reference grid at time tq."""
        tq = float(tq)
        idx = np.searchsorted(self.t, tq)
        idx = min(max(idx, 1), self.t.size - 1)
        if self.t[idx] == tq:
            return self.x, self.U[idx].copy()
        # Linear interpolation between the two bracketing time frames.
        t0, t1 = self.t[idx - 1], self.t[idx]
        w = (tq - t0) / (t1 - t0)
        return self.x, (1.0 - w) * self.U[idx - 1] + w * self.U[idx]

    def save(self, path):
        err = np.array(
            self.error_estimate if self.error_estimate is not None else np.nan
        )
        np.savez_compressed(
            path,
            x=self.x,
            t=self.t,
            U=self.U,
            kind=self.kind,
            nu=self.nu,
            method=self.method,
            error_estimate=err,
        )

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        err = float(d["error_estimate"]) if "error_estimate" in d else None
        if err is not None and np.isnan(err):
            err = None
        return cls(
            d["x"],
            d["t"],
            d["U"],
            kind=str(d["kind"]),
            nu=float(d["nu"]),
            method=int(d["method"]),
            error_estimate=err,
        )


def build_reference(
    kind="linear",
    nu=0.012,
    dx=PI / 1600,
    t_end=PI,
    dtdx=0.5,
    method=1,
    boundary_fn=None,
    ic_fn=None,
    diss_flag=False,
    eps=-0.02,
    max_frames=4000,
):
    """Run the limited finite-volume scheme on a fine grid as a reference.

    Parameters
    ----------
    kind, nu, dtdx : forwarded to convdiff1d.solve
    dx : reference grid spacing (should be much finer than the study grid)
    t_end : final time
    max_frames : cap on stored time frames (subsampled uniformly)
    """
    dt = dtdx * dx
    nsteps = int(round(t_end / dt))
    save_every = max(1, nsteps // max_frames)
    store_steps = sorted(set(range(1, nsteps + 1, save_every)) | {nsteps})

    res = convdiff1d.solve(
        kind=kind,
        method=method,
        nu=nu,
        dx=dx,
        nsteps=nsteps,
        dtdx=dtdx,
        boundary_fn=boundary_fn,
        ic_fn=ic_fn,
        diss_flag=diss_flag,
        eps=eps,
        snapshots=store_steps,
    )

    t = np.array([0.0] + [s * dt for s in store_steps])
    U = np.vstack([res["u_init"]] + [res["snapshots"][s] for s in store_steps])
    return ReferenceSolution(res["x"], t, U, kind=kind, nu=nu, method=method)


def richardson_reference(
    kind="linear",
    nu=0.012,
    t_end=PI,
    dx=PI / 800,
    dtdx=0.5,
    method=1,
    p=2.0,
    boundary_fn=None,
    ic_fn=None,
    diss_flag=False,
    eps=-0.02,
    max_frames=4000,
):
    """Richardson-extrapolated reference plus an L1 field error estimate.

    Solves on dx and dx/2 (same scheme), interpolates the coarse solution onto
    the fine grid, and combines them as
        u_ref = u_fine + (u_fine - u_coarse) / (2**p - 1).
    Returns (ReferenceSolution, error_estimate). The estimate is the L1 mean
    of the leading-order correction term; it is a robust measure of the
    reference error (the pointwise max is dominated by the moving
    discontinuity and is not representative).
    """
    coarse = build_reference(
        kind=kind,
        nu=nu,
        dx=dx,
        t_end=t_end,
        dtdx=dtdx,
        method=method,
        boundary_fn=boundary_fn,
        ic_fn=ic_fn,
        diss_flag=diss_flag,
        eps=eps,
        max_frames=max_frames,
    )
    fine = build_reference(
        kind=kind,
        nu=nu,
        dx=dx / 2.0,
        t_end=t_end,
        dtdx=dtdx,
        method=method,
        boundary_fn=boundary_fn,
        ic_fn=ic_fn,
        diss_flag=diss_flag,
        eps=eps,
        max_frames=max_frames,
    )

    # Interpolate the coarse field onto the fine space-time grid.
    Tf, Xf = np.meshgrid(fine.t, fine.x, indexing="ij")
    Uc_on_fine = coarse.u(Xf, Tf)
    correction = (fine.U - Uc_on_fine) / (2.0**p - 1.0)
    U_ref = fine.U + correction
    U_ref[0] = fine.U[0]
    err = float(np.mean(np.abs(correction[1:])))

    ref = ReferenceSolution(
        fine.x, fine.t, U_ref, kind=kind, nu=nu, method=method, error_estimate=err
    )
    return ref, err


def richardson_reference3(
    kind="linear",
    nu=0.012,
    t_end=PI,
    dx=PI / 1600,
    dtdx=0.5,
    method=1,
    boundary_fn=None,
    ic_fn=None,
    diss_flag=False,
    eps=-0.02,
    max_frames=4000,
    estimate=True,
):
    """Three-grid Richardson reference for a front-limited scheme.

    The limited scheme is only first order across the moving discontinuity, so
    a two-grid combination with ``p=2`` leaves the leading ``O(dx)`` front term
    in the reference. Combining three grids ``(dx, 2dx, 4dx)`` cancels both the
    ``O(dx)`` and ``O(dx^2)`` terms::

        u_ref = u1 + 2 (u1 - u2) - (u1 - u4) / 3,

    with ``u1`` the finest solution and ``u2``/``u4`` the coarser ones
    interpolated onto the fine grid. The combination is applied only for
    ``t > 0``; at ``t = 0`` the exact initial datum is imposed, because
    extrapolating across the discontinuous datum produces nonphysical
    pointwise values. Returns ``(ReferenceSolution, err)``. With
    ``estimate=True`` the residual error is bounded by re-running the same
    combination on ``(dx/2, dx, 2dx)`` and taking the L1 difference over
    ``t > 0``.
    """
    fine = build_reference(
        kind=kind,
        nu=nu,
        dx=dx,
        t_end=t_end,
        dtdx=dtdx,
        method=method,
        boundary_fn=boundary_fn,
        ic_fn=ic_fn,
        diss_flag=diss_flag,
        eps=eps,
        max_frames=max_frames,
    )
    mid = build_reference(
        kind=kind,
        nu=nu,
        dx=2.0 * dx,
        t_end=t_end,
        dtdx=dtdx,
        method=method,
        boundary_fn=boundary_fn,
        ic_fn=ic_fn,
        diss_flag=diss_flag,
        eps=eps,
        max_frames=max_frames,
    )
    coarse = build_reference(
        kind=kind,
        nu=nu,
        dx=4.0 * dx,
        t_end=t_end,
        dtdx=dtdx,
        method=method,
        boundary_fn=boundary_fn,
        ic_fn=ic_fn,
        diss_flag=diss_flag,
        eps=eps,
        max_frames=max_frames,
    )

    Tf, Xf = np.meshgrid(fine.t, fine.x, indexing="ij")
    u1 = fine.U
    u2 = mid.u(Xf, Tf)
    u4 = coarse.u(Xf, Tf)
    U_ref = u1 + 2.0 * (u1 - u2) - (u1 - u4) / 3.0
    U_ref[0] = fine.U[0]

    err = float("nan")
    if estimate:
        finer = build_reference(
            kind=kind,
            nu=nu,
            dx=dx / 2.0,
            t_end=t_end,
            dtdx=dtdx,
            method=method,
            boundary_fn=boundary_fn,
            ic_fn=ic_fn,
            diss_flag=diss_flag,
            eps=eps,
            max_frames=max_frames,
        )
        uf = finer.u(Xf, Tf)
        U_ref_fine = uf + 2.0 * (uf - u1) - (uf - u2) / 3.0
        U_ref_fine[0] = fine.U[0]
        err = float(np.mean(np.abs(U_ref[1:] - U_ref_fine[1:])))

    ref = ReferenceSolution(
        fine.x, fine.t, U_ref, kind=kind, nu=nu, method=method, error_estimate=err
    )
    return ref, err


def exact_linear_nu0(x, t, a=1.0, boundary_fn=None):
    """Closed-form solution for the linear problem with nu = 0."""
    return fv1d.exact_linear(x, t, a=a, boundary_fn=boundary_fn)


def load_or_build(path, **kwargs):
    """Load a cached reference from `path`, building and saving it if absent."""
    import os

    if os.path.exists(path):
        return ReferenceSolution.load(path)
    ref = build_reference(**kwargs)
    ref.save(path)
    return ref

"""Implicit finite-volume solvers for the 1-D convection-diffusion equation.

Direct Python port of the two 1-D methods in final/final_1d.m:

  1. Limited Lax-Wendroff (van Leer MC) for convection, combined with a
     2nd-order central and trapezoidal (Crank-Nicolson) treatment of
     diffusion. The convective flux contributes only to the RHS, so the
     left-hand side is the pure diffusion tridiagonal system.
  2. Trapezoidal in time with 3-point central in space for both convection
     and diffusion (no limiter). Optional 4th-order dissipation term
     eps * dx^3 * u_xxxx treated explicitly.

Both methods solve a tridiagonal system for the interior unknowns. The left
boundary is Dirichlet; the right boundary is closed with a non-reflective
(zero-gradient) condition u(jmax) = u(jmax-1).
"""

import numpy as np

from .fv1d import (
    boundary_oscillatory_8,
    cell_flux,
    initial_condition,
    l1_error,
    exact_linear,
)
from .limiters import minmod, vanleermc


def solve_tridiag(a, b, c, d):
    """Thomas algorithm for a tridiagonal system.

    a, b, c are the sub-, main, and super-diagonal (length n); d is the RHS.
    Ported from the MATLAB function tri in final_1d.m:177-194.
    """
    a = np.asarray(a, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    c = np.asarray(c, dtype=float).copy()
    d = np.asarray(d, dtype=float).copy()
    n = d.size

    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m

    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def solve(
    kind="linear",
    method=1,
    nu=0.012,
    dx=np.pi / 100,
    nsteps=200,
    dtdx=0.5,
    a=1.0,
    xend=np.pi,
    boundary_fn=None,
    ic_fn=None,
    diss_flag=False,
    eps=-0.02,
    snapshots=(),
    limiter="mc",
):
    """Time-march the convection-diffusion equation.

    Parameters
    ----------
    kind : {'linear', 'nonlinear'}
    method : {1, 2}
    nu : diffusivity
    dx : grid spacing
    nsteps : number of time steps N
    dtdx : CFL number dt/dx
    a : linear convection speed
    boundary_fn : callable t -> u(0,t); defaults to the final (cos[8t]) BC
    ic_fn : callable x -> u(x,0); defaults to the discontinuous step
    diss_flag, eps : enable/parametrize the 4th-order dissipation (method 2)
    snapshots : extra step indices to store
    limiter : "mc" (default) or "minmod" for method 1

    Returns
    -------
    dict with the grid and the initial/final/partway fields.
    """
    if boundary_fn is None:
        boundary_fn = boundary_oscillatory_8
    if method not in (1, 2):
        raise ValueError(f"method must be 1 or 2, got {method}")
    if kind not in ("linear", "nonlinear"):
        raise ValueError(f"kind must be 'linear' or 'nonlinear', got {kind!r}")

    jmax = int(round(xend / dx)) + 1
    n = jmax
    dt = dtdx * dx
    x = np.arange(jmax) * dx
    npart = int(round(nsteps / 4))
    store = set(snapshots) | {npart, nsteps}

    u = initial_condition(x) if ic_fn is None else np.asarray(ic_fn(x), dtype=float)
    u_init = u.copy()
    snaps = {}

    for step in range(1, nsteps + 1):
        t = step * dt
        flux, ajac = cell_flux(u, kind, a)

        if method == 1:
            # Limited Lax-Wendroff convective flux.
            dfa = 1.0 - dt / dx * 0.5 * (ajac[:-1] + ajac[1:])
            dfp = flux[1:] - flux[:-1]
            fp = flux.copy()
            fp[0] = flux[0] + 0.5 * dfa[0] * dfp[0]
            limit = vanleermc if limiter == "mc" else minmod
            fp[1 : n - 1] = flux[1 : n - 1] + 0.5 * dfa[1 : n - 1] * limit(
                dfp[1 : n - 1], dfp[0 : n - 2]
            )
            fp[n - 1] = flux[n - 1]

            conv = (fp[1 : n - 1] - fp[0 : n - 2]) / dx
            diff = (u[2:n] - 2.0 * u[1 : n - 1] + u[0 : n - 2]) / dx**2
            rhs = u[1 : n - 1] - dt * conv + 0.5 * nu * dt * diff

            avec = np.full(n - 2, -nu * dt / (2.0 * dx**2))
            bvec = np.full(n - 2, 1.0 + nu * dt / dx**2)
            cvec = np.full(n - 2, -nu * dt / (2.0 * dx**2))

            uL_new = boundary_fn(t)
            avec[0] = 0.0
            rhs[0] += (nu * dt / (2.0 * dx**2)) * uL_new
            bvec[-1] += cvec[-1]
            cvec[-1] = 0.0

        else:
            # Trapezoidal in time, 3-point central in space for conv and diff.
            conv = (flux[2:n] - flux[0 : n - 2]) / (2.0 * dx)
            diff = (u[2:n] - 2.0 * u[1 : n - 1] + u[0 : n - 2]) / dx**2
            rhs = u[1 : n - 1] - 0.5 * dt * conv + 0.5 * nu * dt * diff

            if diss_flag:
                # 4th-order dissipation, treated explicitly (final_1d.m:92-101).
                for k in range(1, n - 3):
                    i = k + 1
                    diss_term = (
                        u[i + 2]
                        - 4.0 * u[i + 1]
                        + 6.0 * u[i]
                        - 4.0 * u[i - 1]
                        + u[i - 2]
                    )
                    rhs[k] += (eps * dt / dx) * diss_term

            avec = -dt / (4.0 * dx) * ajac[0 : n - 2] - nu * dt / (2.0 * dx**2)
            bvec = 1.0 + (nu * dt / dx**2) * np.ones(n - 2)
            cvec = dt / (4.0 * dx) * ajac[2:n] - nu * dt / (2.0 * dx**2)

            uL_new = boundary_fn(t)
            avec[0] = 0.0
            rhs[0] += (dt / (4.0 * dx) * ajac[0] + nu * dt / (2.0 * dx**2)) * uL_new
            bvec[-1] += cvec[-1]
            cvec[-1] = 0.0

        uint_new = solve_tridiag(avec, bvec, cvec, rhs)

        u_new = u.copy()
        u_new[0] = uL_new
        u_new[1 : n - 1] = uint_new
        u_new[n - 1] = u_new[n - 2]
        u = u_new

        if step in store:
            snaps[step] = u.copy()

    result = {
        "kind": kind,
        "method": method,
        "nu": nu,
        "x": x,
        "dx": dx,
        "dt": dt,
        "dtdx": dtdx,
        "nsteps": nsteps,
        "npart": npart,
        "u_init": u_init,
        "u_final": snaps[nsteps],
        "u_part": snaps[npart],
        "snapshots": snaps,
    }
    # The closed-form exact solution only applies to pure convection (nu = 0).
    if kind == "linear" and nu == 0.0:
        result["exact_final"] = exact_linear(x, nsteps * dt, a, boundary_fn)
        result["L1_final"] = l1_error(snaps[nsteps], result["exact_final"])
    return result

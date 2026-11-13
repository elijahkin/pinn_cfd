"""Explicit finite-volume solvers for the linear/nonlinear convection equation.

Direct Python port of hw10/hw10.m. The four methods are:

  1. Euler explicit, 1st-order upwind (backward) in space
  2. Lax-Wendroff (central in space)
  3. Limited Lax-Wendroff (van Leer MC limiter)
  4. RK-3 in time with 2nd-order limited (van Leer MC) in space

The update is always written in flux-difference form,
    u_j^{n+1} = u_j^n - (dt/dx) (f_{j+1/2} - f_{j-1/2}),
with the interface flux f_{j+1/2} chosen per method. The left boundary is a
Dirichlet condition applied through the exact boundary value; the right
boundary is closed with a first-order flux.
"""

import numpy as np

from .limiters import vanleermc


def boundary_oscillatory_12(t):
    """Dirichlet boundary condition from hw10.tex:56 (contains cos[12t])."""
    return 1.0 - 0.25 * np.cos(4.0 * t) - 0.25 * np.cos(12.0 * t)


def boundary_oscillatory_8(t):
    """Dirichlet boundary condition from final.tex:56 (contains cos[8t])."""
    return 1.0 - 0.25 * np.cos(4.0 * t) - 0.25 * np.cos(8.0 * t)


def initial_condition(x):
    """Discontinuous initial condition from hw10.tex:52."""
    x = np.asarray(x, dtype=float)
    return np.where(x < 0.5 * np.pi, 0.5, 1.5)


def cell_flux(u, kind="linear", a=1.0):
    """Return the cell flux f(u) and the flux Jacobian df/du."""
    u = np.asarray(u, dtype=float)
    if kind == "linear":
        return a * u, a * np.ones_like(u)
    if kind == "nonlinear":
        return 0.5 * u * u, u.copy()
    raise ValueError(f"kind must be 'linear' or 'nonlinear', got {kind!r}")


def exact_linear(x, t, a=1.0, boundary_fn=None):
    """Exact solution of the linear problem (hw10.m:184-206).

    The initial step convects right at speed a; points swept by the left
    boundary (x <= a t) follow the oscillatory Dirichlet data.
    """
    if boundary_fn is None:
        boundary_fn = boundary_oscillatory_12
    x = np.asarray(x, dtype=float)
    xi = x - a * t
    ut = np.where(xi <= 0.5 * np.pi, 0.5, 1.5)
    mask = xi <= 0.0
    if np.any(mask):
        s = t - x[mask] / a
        ut[mask] = boundary_fn(s)
    return ut


def l1_error(u, u_exact):
    """Mean absolute error, matching the MATLAB L1norm convention."""
    u = np.asarray(u, dtype=float)
    u_exact = np.asarray(u_exact, dtype=float)
    return float(np.sum(np.abs(u - u_exact)) / u.size)


def _limited_interface_flux(flux, dfp):
    """Second-order limited interface flux used by methods 3 and 4.

    Mirrors hw10.m:90-92 / 102-105: central (LW) at the first interface and
    at the right boundary, van Leer MC limited in the interior.
    """
    n = flux.size
    fp = np.empty(n, dtype=float)
    fp[0] = flux[0] + 0.5 * dfp[0]
    fp[1 : n - 1] = flux[1 : n - 1] + 0.5 * vanleermc(dfp[1 : n - 1], dfp[0 : n - 2])
    fp[n - 1] = flux[n - 1]
    return fp


def solve(
    kind="linear",
    method=1,
    dx=np.pi / 80,
    nsteps=160,
    dtdx=0.5,
    a=1.0,
    xend=np.pi,
    boundary_fn=None,
    snapshots=(),
):
    """Time-march the convection equation with one of the four explicit methods.

    Parameters
    ----------
    kind : {'linear', 'nonlinear'}
    method : {1, 2, 3, 4}
    dx : grid spacing (uniform)
    nsteps : number of time steps N
    dtdx : CFL number dt/dx
    a : linear convection speed
    xend : right end of the domain
    boundary_fn : callable t -> u(0,t); defaults to the hw10 (cos[12t]) BC
    snapshots : extra step indices to store

    Returns
    -------
    dict with the grid, the initial/final/partway fields, and (for the linear
    problem) the exact fields and the final L1 error.
    """
    if boundary_fn is None:
        boundary_fn = boundary_oscillatory_12
    if method not in (1, 2, 3, 4):
        raise ValueError(f"method must be 1-4, got {method}")
    if kind not in ("linear", "nonlinear"):
        raise ValueError(f"kind must be 'linear' or 'nonlinear', got {kind!r}")

    jmax = int(round(xend / dx)) + 1
    dt = dtdx * dx
    x = np.arange(jmax) * dx
    npart = int(round(nsteps / 4))
    store = set(snapshots) | {npart, nsteps}

    u = initial_condition(x)
    u_init = u.copy()

    snaps = {}
    exact_snaps = {}

    for n in range(1, nsteps + 1):
        t = n * dt
        u1old = boundary_fn(t - dt)
        u1 = boundary_fn(t)

        flux, ajac = cell_flux(u, kind, a)

        if method == 1:
            # Euler explicit with first-order backward (upwind) flux.
            fp = flux.copy()
            du = np.empty_like(u)
            du[0] = u1 - u1old
            du[1:] = -dt / dx * (fp[1:] - fp[:-1])

        elif method == 2:
            # Lax-Wendroff: f_{j+1/2} = f_j + 0.5*(1 - c*a_{j+1/2})(f_{j+1}-f_j)
            dfa = 1.0 - dt / dx * 0.5 * (ajac[:-1] + ajac[1:])
            dfp = flux[1:] - flux[:-1]
            fp = flux.copy()
            fp[0] = flux[0] + 0.5 * dfa[0] * dfp[0]
            fp[1 : jmax - 1] = (
                flux[1 : jmax - 1] + 0.5 * dfa[1 : jmax - 1] * dfp[1 : jmax - 1]
            )
            fp[jmax - 1] = flux[jmax - 1]
            du = np.empty_like(u)
            du[0] = u1 - u1old
            du[1:] = -dt / dx * (fp[1:] - fp[:-1])

        elif method == 3:
            # Limited Lax-Wendroff (van Leer MC).
            dfa = 1.0 - dt / dx * 0.5 * (ajac[:-1] + ajac[1:])
            dfp = flux[1:] - flux[:-1]
            fp = flux.copy()
            fp[0] = flux[0] + 0.5 * dfa[0] * dfp[0]
            fp[1 : jmax - 1] = flux[1 : jmax - 1] + 0.5 * dfa[1 : jmax - 1] * vanleermc(
                dfp[1 : jmax - 1], dfp[0 : jmax - 2]
            )
            fp[jmax - 1] = flux[jmax - 1]
            du = np.empty_like(u)
            du[0] = u1 - u1old
            du[1:] = -dt / dx * (fp[1:] - fp[:-1])

        else:
            # RK-3 in time, limited (van Leer MC) 2nd-order in space.
            dfp = flux[1:] - flux[:-1]
            fp = _limited_interface_flux(flux, dfp)
            du = np.empty_like(u)
            du[0] = u1 - u1old
            du[1:] = -dt / dx * (fp[1:] - fp[:-1])
            upred1 = u + du

            flux, _ = cell_flux(upred1, kind, a)
            dfp = flux[1:] - flux[:-1]
            fpp = _limited_interface_flux(flux, dfp)
            du = np.empty_like(u)
            du[0] = u1 - u1old
            du[1:] = -dt / dx * (fpp[1:] - fpp[:-1])
            upred2 = 0.75 * u + 0.25 * upred1 + 0.25 * du

            flux, _ = cell_flux(upred2, kind, a)
            dfp = flux[1:] - flux[:-1]
            fpp2 = _limited_interface_flux(flux, dfp)
            du = np.empty_like(u)
            du[0] = u1 - u1old
            du[1:] = -dt / dx * (fpp2[1:] - fpp2[:-1])
            du = (1.0 / 3.0) * u + (2.0 / 3.0) * upred2 + (2.0 / 3.0) * du - u

        u = u + du

        if n in store:
            snaps[n] = u.copy()
            if kind == "linear":
                exact_snaps[n] = exact_linear(x, t, a, boundary_fn)

    result = {
        "kind": kind,
        "method": method,
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
    if kind == "linear":
        result["exact_final"] = exact_snaps[nsteps]
        result["exact_part"] = exact_snaps[npart]
        result["L1_final"] = l1_error(snaps[nsteps], exact_snaps[nsteps])
    return result

"""Validation of the reference-solution module.

Run from the project root:

    python -m pinn_cfd.experiments.validate_reference

Checks:
  A. nu = 0: the fine-grid reference agrees with the closed-form solution.
  B. Self-convergence of the fine-grid reference sequence (linear, nu > 0).
  C. Richardson reference: error estimate is consistent with a finer solve.
  D. Independent cross-check: limited vs. central scheme on a fine grid.
  E. Interpolator reproduces the stored field and the nonlinear reference is
     finite.
"""

import numpy as np

from pinn_cfd.classic import convdiff1d, fv1d, reference
from pinn_cfd.classic.fv1d import l1_error

PI = np.pi


def check_nu0():
    print("A. nu=0 reference vs closed-form solution")
    t_end = PI
    bc = fv1d.boundary_oscillatory_8
    ref = reference.build_reference(
        kind="linear", nu=0.0, dx=PI / 1600, t_end=t_end, boundary_fn=bc
    )
    x = ref.x
    ue = reference.exact_linear_nu0(x, t_end, boundary_fn=bc)
    err = l1_error(ref.field(t_end)[1], ue)
    print(f"   L1 error vs exact at t=pi: {err:.3e}")
    return err < 5e-3


def check_self_convergence():
    print("\nB. Fine-grid self-convergence (linear, nu=0.012)")
    t_end = PI
    dxs = [PI / 100, PI / 200, PI / 400, PI / 800]
    refs = [
        reference.build_reference(kind="linear", nu=0.012, dx=dx, t_end=t_end)
        for dx in dxs
    ]
    x_finest = refs[-1].x
    finest = refs[-1].field(t_end)[1]
    errs = np.array([l1_error(r.u(x_finest, t_end), finest) for r in refs[:-1]])
    order = np.polyfit(np.log(dxs[:-1]), np.log(errs), 1)[0]
    print("   L1 vs finest: " + "  ".join(f"{e:.3e}" for e in errs))
    print(f"   observed order = {order:.2f}")
    return order > 0.8


def check_richardson():
    print("\nC. Richardson reference (linear, nu=0.012)")
    t_end = PI
    ref, err_est = reference.richardson_reference(
        kind="linear", nu=0.012, t_end=t_end, dx=PI / 400, p=2.0
    )
    # Independent finer single-grid solve to test the error estimate.
    fine = reference.build_reference(kind="linear", nu=0.012, dx=PI / 1600, t_end=t_end)
    _, u_fine = fine.field(t_end)
    u_ref = ref.u(fine.x, t_end)
    actual = l1_error(u_ref, u_fine)
    print(f"   Richardson L1 correction (error estimate) = {err_est:.3e}")
    print(f"   |Richardson - fine(pi/1600)| L1            = {actual:.3e}")
    return actual < 5 * err_est + 1e-4


def check_cross_scheme():
    print("\nD. Independent scheme cross-check (linear)")
    t_end = PI
    dx = PI / 800
    for nu in (0.012, 0.004, 0.002):
        m1 = reference.build_reference(
            kind="linear", nu=nu, dx=dx, t_end=t_end, method=1
        )
        m2 = reference.build_reference(
            kind="linear", nu=nu, dx=dx, t_end=t_end, method=2, diss_flag=True
        )
        _, u1 = m1.field(t_end)
        _, u2 = m2.field(t_end)
        print(f"   nu={nu:<6}: L1(method1 - method2+diss) = {l1_error(u1, u2):.3e}")


def check_interp_and_nonlinear():
    print("\nE. Interpolator and nonlinear reference")
    ref = reference.build_reference(kind="linear", nu=0.012, dx=PI / 400, t_end=PI)
    i = 7
    xg, tg = ref.x, ref.t[i]
    err = np.max(np.abs(ref.u(xg, np.full_like(xg, tg)) - ref.U[i]))
    print(f"   max|interp(grid) - stored| = {err:.3e}")

    nl = reference.build_reference(kind="nonlinear", nu=0.004, dx=PI / 800, t_end=PI)
    finite = np.all(np.isfinite(nl.U))
    print(
        f"   nonlinear reference finite: {finite}, "
        f"range = [{nl.U.min():.3f}, {nl.U.max():.3f}]"
    )
    return err < 1e-12 and finite


def main():
    ok = check_nu0()
    ok = check_self_convergence() and ok
    ok = check_richardson() and ok
    check_cross_scheme()
    ok = check_interp_and_nonlinear() and ok
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

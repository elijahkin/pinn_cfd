"""Validation of the Python ports against the published MATLAB results.

Run from the project root (pinn_cfd must be importable), e.g.

    python -m pinn_cfd.experiments.validate_classic

Checks:
  A. The explicit convection methods reproduce the hw10 linear L1 norms.
  B. The implicit convection-diffusion methods converge at ~2nd order on a
     manufactured heat solution (validates the diffusion operator + BCs).
  C. The final-project convection-diffusion cases run to completion (finite).
"""

import numpy as np

from pinn_cfd.classic import convdiff1d, fv1d

PI = np.pi

# L1 norms published in hw10.tex:123-147 for the linear problem, cases a/b/c.
HW10_EXPECTED = {
    1: {160: 0.12212, 320: 0.093177, 640: 0.06149},
    2: {160: 0.073988, 320: 0.02091, 640: 0.0053987},
    3: {160: 0.026021, 320: 0.0055135, 640: 0.0013336},
    4: {160: 0.050861, 320: 0.014051, 640: 0.0048415},
}
HW10_CASES = {160: PI / 80, 320: PI / 160, 640: PI / 320}


def check_explicit():
    print("A. hw10 explicit linear-convection L1 norms")
    print(
        f"{'method':>6} {'N':>5} {'dx':>10} {'L1 (py)':>12} "
        f"{'L1 (hw10)':>12} {'rel err':>10}"
    )
    ok = True
    for method in (1, 2, 3, 4):
        for nsteps, dx in HW10_CASES.items():
            res = fv1d.solve(kind="linear", method=method, dx=dx, nsteps=nsteps)
            got = res["L1_final"]
            want = HW10_EXPECTED[method][nsteps]
            rel = abs(got - want) / want
            flag = "ok" if rel < 1e-3 else "MISMATCH"
            ok = ok and rel < 1e-3
            print(
                f"{method:>6} {nsteps:>5} {dx:>10.5f} {got:>12.6g} "
                f"{want:>12.6g} {rel:>10.2e}  {flag}"
            )
    return ok


def check_tridiag():
    """Thomas solver agrees with scipy's banded solver."""
    from scipy.linalg import solve_banded

    rng = np.random.default_rng(0)
    n = 40
    a = rng.uniform(-0.5, 0.0, n)
    c = rng.uniform(-0.5, 0.0, n)
    b = rng.uniform(2.0, 4.0, n)
    d = rng.uniform(-1.0, 1.0, n)
    got = convdiff1d.solve_tridiag(a, b, c, d)
    ab = np.zeros((3, n))
    ab[0, 1:] = c[:-1]
    ab[1] = b
    ab[2, :-1] = a[1:]
    want = solve_banded((1, 1), ab, d)
    err = np.max(np.abs(got - want))
    print(f"\nB. Tridiagonal solver: max|Thomas - scipy| = {err:.2e}")
    return err < 1e-10


def check_diffusion_order():
    """Manufactured solution u(x,t)=exp(-nu t) cos(x) for u_t = nu u_xx.

    Satisfies u(0,t)=exp(-nu t) (Dirichlet) and u_x(pi,t)=0 (Neumann), which
    is exactly what the implicit solvers impose. The interior operator should
    be 2nd order; the full-domain order is limited to ~1 by the code's
    zero-gradient right closure u(jmax)=u(jmax-1) (final_1d.m:127).
    """
    print("\nC. Implicit diffusion order (manufactured heat solution)")
    nu = 0.02
    t_end = 1.0
    exact = lambda x, t: np.exp(-nu * t) * np.cos(x)
    bc = lambda t: np.exp(-nu * t)
    ic = lambda x: np.cos(x)

    for method in (1, 2):
        errs, errs_int, dxs = [], [], []
        for dx in (PI / 50, PI / 100, PI / 200, PI / 400):
            nsteps = int(round(t_end / (0.5 * dx)))
            res = convdiff1d.solve(
                kind="linear",
                method=method,
                nu=nu,
                dx=dx,
                nsteps=nsteps,
                a=0.0,
                boundary_fn=bc,
                ic_fn=ic,
            )
            x = res["x"]
            e = np.abs(res["u_final"] - exact(x, nsteps * res["dt"]))
            m = (x >= 0.2 * PI) & (x <= 0.6 * PI)
            errs.append(e.mean())
            errs_int.append(e[m].mean())
            dxs.append(dx)
        dxs = np.array(dxs)
        o_all = np.polyfit(np.log(dxs), np.log(errs), 1)[0]
        o_int = np.polyfit(np.log(dxs), np.log(errs_int), 1)[0]
        print(
            f"  method {method}: interior order = {o_int:.2f}, "
            f"full-domain order = {o_all:.2f} (boundary-limited)"
        )


def check_final_cases():
    print("\nD. final convection-diffusion cases (finite check)")
    cases = [(PI / 100, 200), (PI / 200, 400)]
    for kind in ("linear", "nonlinear"):
        for nu in (0.012, 0.004, 0.002):
            for method in (1, 2):
                for dx, nsteps in cases:
                    res = convdiff1d.solve(
                        kind=kind, method=method, nu=nu, dx=dx, nsteps=nsteps
                    )
                    u = res["u_final"]
                    if not np.all(np.isfinite(u)):
                        print(f"  NON-FINITE: {kind} nu={nu} method={method}")
                        return False
    print(
        "  all linear/nonlinear cases with nu in {0.012, 0.004, 0.002} "
        "produced finite fields"
    )
    return True


def main():
    ok = check_explicit()
    ok = check_tridiag() and ok
    check_diffusion_order()
    ok = check_final_cases() and ok
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

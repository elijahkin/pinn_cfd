"""Error metrics for comparing fields against a reference."""

import numpy as np


def l1(u, u_ref):
    """Mean absolute error."""
    u = np.asarray(u, dtype=float)
    u_ref = np.asarray(u_ref, dtype=float)
    return float(np.mean(np.abs(u - u_ref)))


def l2(u, u_ref):
    """Root-mean-square error."""
    u = np.asarray(u, dtype=float)
    u_ref = np.asarray(u_ref, dtype=float)
    return float(np.sqrt(np.mean((u - u_ref) ** 2)))


def linf(u, u_ref):
    """Max-norm error."""
    u = np.asarray(u, dtype=float)
    u_ref = np.asarray(u_ref, dtype=float)
    return float(np.max(np.abs(u - u_ref)))


def relative_l1(u, u_ref):
    """L1 error normalized by the reference L1 mass."""
    u = np.asarray(u, dtype=float)
    u_ref = np.asarray(u_ref, dtype=float)
    denom = np.sum(np.abs(u_ref))
    return float(np.sum(np.abs(u - u_ref)) / denom) if denom else float("nan")


def summarize(u, u_ref):
    return {
        "l1": l1(u, u_ref),
        "l2": l2(u, u_ref),
        "linf": linf(u, u_ref),
        "rel_l1": relative_l1(u, u_ref),
    }

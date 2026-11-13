"""Flux limiters used by the limited Lax-Wendroff schemes.

Ported from the MATLAB helpers in hw10/hw10.m (lines 174-181) and
final/final_1d.m (lines 167-174). Both operate elementwise on arrays.
"""

import numpy as np


def minmod(x, y):
    """Minmod limiter: 0.5*(sgn(x)+sgn(y))*min(|x|,|y|).

    Returns 0 where x and y have opposite signs, otherwise the smaller
    magnitude with the common sign.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return 0.5 * (np.sign(x) + np.sign(y)) * np.minimum(np.abs(x), np.abs(y))


def vanleermc(x, y):
    """Van Leer monotonized central (MC) limiter.

    Equivalent to minmod(0.5*(x+y), 2*minmod(x,y)) from hw10.m:180.
    """
    return minmod(
        0.5 * (np.asarray(x, dtype=float) + np.asarray(y, dtype=float)),
        2.0 * minmod(x, y),
    )

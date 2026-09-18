#!/usr/bin/env python3
import numpy as np
import run_robustness_validation as rv


def stable_irr_monthly(returns):
    bal = rv.INIT
    for r in returns:
        bal += rv.MONTHLY
        bal *= 1 + r
    n = len(returns)

    def fv_gap(mr):
        g = 1.0 + mr
        init_fv = rv.INIT * (g ** n)
        if abs(mr) < 1e-12:
            contrib_fv = rv.MONTHLY * n
        else:
            contrib_fv = rv.MONTHLY * g * ((g ** n) - 1.0) / mr
        return init_fv + contrib_fv - bal

    lo, hi = -0.50, 1.0
    flo, fhi = fv_gap(lo), fv_gap(hi)
    for _ in range(100):
        if np.sign(flo) != np.sign(fhi):
            break
        hi *= 2.0
        fhi = fv_gap(hi)
    if np.sign(flo) == np.sign(fhi):
        return bal, np.nan
    for _ in range(120):
        mid = (lo + hi) / 2.0
        fm = fv_gap(mid)
        if abs(fm) < 1e-8:
            lo = hi = mid
            break
        if np.sign(fm) == np.sign(flo):
            lo, flo = mid, fm
        else:
            hi = mid
    mr = (lo + hi) / 2.0
    return bal, (1.0 + mr) ** 12 - 1.0


rv.irr_monthly = stable_irr_monthly
rv.main()

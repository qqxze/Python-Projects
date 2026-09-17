#!/usr/bin/env python3
"""Compatibility entrypoint that isolates official FinRL-X's `src` package.

The local V9 harness also has a package named `src`; importing the V9 strategy first
loads that package. Before importing official FinRL-X BacktestEngine, purge only the
`src` namespace and put the pinned FinRL-X checkout first on sys.path.
"""
from __future__ import annotations

import os
import sys
import pandas as pd

import run_finrl_real_validation as m


def patched_run_finrl_engine(name: str, engine_prices: pd.DataFrame, weight_path: pd.DataFrame):
    root = os.environ.get("FINRL_X_ROOT")
    if not root:
        raise RuntimeError("FINRL_X_ROOT is required")

    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            del sys.modules[key]
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)

    from src.backtest.backtest_engine import BacktestEngine, BacktestConfig

    cfg = BacktestConfig(
        start_date=m.REAL_START,
        end_date=m.REAL_END,
        initial_capital=m.INIT,
        transaction_cost=m.TC,
        benchmark_tickers=["QQQ", "SPY"],
        integer_positions=False,
    )
    eng = BacktestEngine(cfg)
    result = eng.run_backtest(name, engine_prices, weight_path)
    values = result.portfolio_values.copy()
    values.index = pd.to_datetime(values.index).tz_localize(None)
    return result, values.loc[m.REAL_START:m.REAL_END]


m.run_finrl_engine = patched_run_finrl_engine
m.main()

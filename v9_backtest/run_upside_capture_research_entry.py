#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import pandas as pd

import run_finrl_real_validation as real


def patched(name: str, engine_prices: pd.DataFrame, weight_path: pd.DataFrame):
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
        start_date=real.REAL_START,
        end_date=real.REAL_END,
        initial_capital=real.INIT,
        transaction_cost=real.TC,
        benchmark_tickers=["QQQ", "SPY"],
        integer_positions=False,
    )
    eng = BacktestEngine(cfg)
    result = eng.run_backtest(name, engine_prices, weight_path)
    values = result.portfolio_values.copy()
    values.index = pd.to_datetime(values.index).tz_localize(None)
    return result, values.loc[real.REAL_START:real.REAL_END]


real.run_finrl_engine = patched
import run_upside_capture_research as research
research.main()

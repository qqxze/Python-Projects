#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Local frozen V9 implementation / production execution helpers.
import run_frequency_backtest as rf
import run_v10_research_experiments as r1
import run_v10_round2 as r2

FINRL_COMMIT = "e65d6f0483ead7d2ef4a5fc940cdf960392a25c1"
WARMUP_START = "2008-01-02"
REAL_START = "2010-02-11"  # TQQQ inception; frozen V9 core does not use TQQQ, but we audit it.
REAL_END = "2025-12-31"
YF_END_EXCLUSIVE = "2026-01-02"
TC = 0.001
INIT = 1_000_000.0
MONTHLY = 20_000.0
ASSETS_ENGINE = ["QQQ", "SPY", "QLD", "SSO", "BIL"]
DOWNLOAD_TICKERS = ["QQQ", "SPY", "QLD", "SSO", "TQQQ", "BIL"]
OUT = Path("finrl_real_validation")
OUT.mkdir(exist_ok=True)


def download_one(ticker: str) -> pd.Series:
    last_err = None
    for attempt in range(4):
        try:
            d = yf.download(
                ticker,
                start=WARMUP_START,
                end=YF_END_EXCLUSIVE,
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )
            if d is None or d.empty:
                raise RuntimeError(f"empty Yahoo data for {ticker}")
            if isinstance(d.columns, pd.MultiIndex):
                # yfinance may return (field,ticker) even for one ticker.
                if ("Adj Close", ticker) in d.columns:
                    s = d[("Adj Close", ticker)]
                elif ("Close", ticker) in d.columns:
                    s = d[("Close", ticker)]
                else:
                    s = d.xs(ticker, axis=1, level=-1).get("Adj Close", d.xs(ticker, axis=1, level=-1)["Close"])
            else:
                s = d["Adj Close"] if "Adj Close" in d.columns else d["Close"]
            s = pd.to_numeric(s, errors="coerce").dropna().astype(float)
            s.index = pd.to_datetime(s.index).tz_localize(None)
            s.name = ticker
            if len(s) < 1000:
                raise RuntimeError(f"too few rows for {ticker}: {len(s)}")
            return s
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Yahoo download failed for {ticker}: {last_err}")


def load_real_prices() -> tuple[pd.DataFrame, dict]:
    series = {t: download_one(t) for t in DOWNLOAD_TICKERS}
    raw = pd.concat(series.values(), axis=1).sort_index()
    raw = raw.loc[WARMUP_START:REAL_END]

    audit = {"source": "Yahoo Finance via yfinance", "auto_adjust": False, "price_field": "Adj Close"}
    for t, s in series.items():
        ss = s.loc[:REAL_END]
        audit[t] = {
            "first": str(ss.first_valid_index().date()),
            "last": str(ss.last_valid_index().date()),
            "rows": int(ss.notna().sum()),
        }

    # Strategy features only use QQQ / SPY / CASH. QLD/SSO are real since 2006.
    # TQQQ is intentionally disabled in frozen V9 1.34x, so its pre-inception NaNs
    # never enter the signal or return path. CASH is represented by real BIL Adj Close.
    full = pd.DataFrame(index=raw.index)
    full["QQQ"] = raw["QQQ"]
    full["SPY"] = raw["SPY"]
    full["QLD"] = raw["QLD"]
    full["SSO"] = raw["SSO"]
    full["TQQQ"] = raw["TQQQ"]
    full["CASH"] = raw["BIL"]

    # Use the common liquid calendar for the assets that frozen V9 can actually hold.
    common = raw[["QQQ", "SPY", "QLD", "SSO", "BIL"]].dropna().index
    full = full.reindex(common)
    audit["common_real_start_core_assets"] = str(common.min().date())
    audit["backtest_start"] = REAL_START
    audit["backtest_end"] = REAL_END
    audit["no_synthetic_execution_prices"] = True
    audit["cash_proxy"] = "BIL real adjusted-close total-return proxy"
    audit["TQQQ_note"] = "Real TQQQ downloaded/audited from inception; frozen V9 1.34x sets TQQQ weight to zero."
    return full, audit


def metrics_from_values(v: pd.Series) -> dict:
    v = v.dropna().astype(float)
    r = v.pct_change().dropna()
    yrs = (v.index[-1] - v.index[0]).days / 365.2425
    cagr = float((v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1)
    dd = v / v.cummax() - 1
    sd = float(r.std(ddof=1))
    return {
        "start_value": float(v.iloc[0]),
        "final_value": float(v.iloc[-1]),
        "total_return": float(v.iloc[-1] / v.iloc[0] - 1),
        "cagr": cagr,
        "max_drawdown": float(dd.min()),
        "sharpe": float(r.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan,
        "calmar": float(cagr / abs(dd.min())) if dd.min() < 0 else np.nan,
    }


def map_to_engine_weights(w: pd.Series) -> pd.Series:
    out = pd.Series(0.0, index=ASSETS_ENGINE)
    for a in ["QQQ", "SPY", "QLD", "SSO"]:
        out[a] = float(w.get(a, 0.0))
    out["BIL"] = float(w.get("CASH", 0.0))
    # Frozen V9 baseline must never use TQQQ.
    if abs(float(w.get("TQQQ", 0.0))) > 1e-12:
        raise AssertionError("Frozen V9 1.34x unexpectedly produced non-zero TQQQ weight")
    s = float(out.sum())
    return out / s if s > 0 else pd.Series({"BIL": 1.0, **{a: 0.0 for a in ASSETS_ENGINE if a != "BIL"}})


def build_drift_weight_path(engine_prices: pd.DataFrame, full_px: pd.DataFrame, targets: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    # Full monthly schedule is needed so we can recover the actual target on each
    # soft-rebalance trade date and the last known target before REAL_START.
    schedule = rf.make_schedule(full_px, targets, "monthly")
    start = pd.Timestamp(REAL_START)
    idx = engine_prices.index

    prior = schedule.loc[:start]
    if prior.empty:
        raise RuntimeError("No pre-start V9 monthly decision available; warmup is insufficient")
    initial_target = map_to_engine_weights(prior.iloc[-1])

    reset = {start: initial_target}
    if trades is not None and len(trades):
        for d in pd.to_datetime(trades["date"]):
            if d < start or d > idx[-1] or d not in schedule.index:
                continue
            reset[pd.Timestamp(d)] = map_to_engine_weights(schedule.loc[d])

    path = pd.DataFrame(index=idx, columns=ASSETS_ENGINE, dtype=float)
    anchor_w = initial_target.copy()
    anchor_p = engine_prices.loc[idx[0], ASSETS_ENGINE].astype(float)
    for d in idx:
        if d in reset:
            anchor_w = reset[d].copy()
            anchor_p = engine_prices.loc[d, ASSETS_ENGINE].astype(float)
            path.loc[d] = anchor_w
        else:
            rel = engine_prices.loc[d, ASSETS_ENGINE].astype(float) / anchor_p
            w = anchor_w * rel
            path.loc[d] = w / float(w.sum())
    return path.astype(float)


def run_finrl_engine(name: str, engine_prices: pd.DataFrame, weight_path: pd.DataFrame):
    root = os.environ.get("FINRL_X_ROOT")
    if not root:
        raise RuntimeError("FINRL_X_ROOT is required")
    sys.path.insert(0, root)
    from src.backtest.backtest_engine import BacktestEngine, BacktestConfig

    cfg = BacktestConfig(
        start_date=REAL_START,
        end_date=REAL_END,
        initial_capital=INIT,
        transaction_cost=TC,
        benchmark_tickers=["QQQ", "SPY"],
        integer_positions=False,
    )
    eng = BacktestEngine(cfg)
    result = eng.run_backtest(name, engine_prices, weight_path)
    values = result.portfolio_values.copy()
    values.index = pd.to_datetime(values.index).tz_localize(None)
    return result, values.loc[REAL_START:REAL_END]


def main():
    full_px, data_audit = load_real_prices()
    if pd.Timestamp(REAL_START) < full_px.index.min():
        raise RuntimeError("Real start precedes common core-asset history")

    # Freeze production period while retaining pre-start warmup for features.
    old_start, old_end = rf.START, rf.END
    rf.START, rf.END = REAL_START, REAL_END
    try:
        base_targets, base_meta = rf.strategy(full_px)
        credit = r1.load_credit(Path("data/hy_oas_full.csv"), full_px.index)
        meta = r1.add_research_features(full_px, base_meta, credit)
        credit_targets, credit_meta = r2.make_targets_round2(
            base_targets, meta, bull2=False, credit_gate=True, growth=False
        )

        base_acc, base_tr, base_fl, base_diag = rf.simulate(full_px, base_targets, meta, "monthly")
        cred_acc, cred_tr, cred_fl, cred_diag = rf.simulate(full_px, credit_targets, credit_meta, "monthly")
        qqq_acc = rf.run_qqq(full_px)
    finally:
        rf.START, rf.END = old_start, old_end

    # Official FinRL-X engine uses a bt target-weight backtester and internally
    # expands signals to daily. Feed the natural drift weight path so daily calls
    # are no-trade between our actual monthly soft-rebalance reset dates.
    ep = pd.DataFrame(index=full_px.loc[REAL_START:REAL_END].index)
    ep["QQQ"] = full_px.loc[REAL_START:REAL_END, "QQQ"]
    ep["SPY"] = full_px.loc[REAL_START:REAL_END, "SPY"]
    ep["QLD"] = full_px.loc[REAL_START:REAL_END, "QLD"]
    ep["SSO"] = full_px.loc[REAL_START:REAL_END, "SSO"]
    ep["BIL"] = full_px.loc[REAL_START:REAL_END, "CASH"]
    ep = ep.dropna()

    base_path = build_drift_weight_path(ep, full_px, base_targets, base_tr)
    cred_path = build_drift_weight_path(ep, full_px, credit_targets, cred_tr)

    base_result, base_values = run_finrl_engine("V9_Real", ep, base_path)
    cred_result, cred_values = run_finrl_engine("V9_1_CreditGate_Real", ep, cred_path)

    # Real QQQ buy/hold unitized benchmark over exactly the same dates.
    q = ep.loc[base_values.index, "QQQ"]
    q_values = INIT * q / q.iloc[0]

    finrl_rows = []
    for name, vals in [("V9", base_values), ("V9.1_CreditGate", cred_values), ("QQQ_BuyHold", q_values)]:
        finrl_rows.append({"variant": name, **metrics_from_values(vals)})
    finrl_df = pd.DataFrame(finrl_rows)
    finrl_df.to_csv(OUT / "finrl_unitized_summary.csv", index=False)

    dca_rows = [
        {"variant": "V9", **rf.perf(base_acc, base_diag)},
        {"variant": "V9.1_CreditGate", **rf.perf(cred_acc, cred_diag)},
        {"variant": "QQQ_DCA", **rf.perf(qqq_acc, {"decision_count":0,"rebalance_count":0,"annual_turnover_notional_to_avg_nav":0.0,"total_cost":float(qqq_acc.cost.sum())})},
    ]
    dca_df = pd.DataFrame(dca_rows)
    dca_df["total_invested"] = INIT + MONTHLY * len(base_acc.index.to_period("M").unique())
    dca_df.to_csv(OUT / "dca_summary.csv", index=False)

    base_acc.to_csv(OUT / "dca_account_v9.csv")
    cred_acc.to_csv(OUT / "dca_account_v9_1_credit.csv")
    qqq_acc.to_csv(OUT / "dca_account_qqq.csv")
    base_tr.to_csv(OUT / "soft_rebalance_trades_v9.csv", index=False)
    cred_tr.to_csv(OUT / "soft_rebalance_trades_v9_1_credit.csv", index=False)
    base_path.to_csv(OUT / "finrl_weight_path_v9.csv")
    cred_path.to_csv(OUT / "finrl_weight_path_v9_1_credit.csv")
    pd.DataFrame({"V9": base_values, "V9.1_CreditGate": cred_values, "QQQ_BuyHold": q_values}).to_csv(OUT / "finrl_nav.csv")

    credit_md = meta.loc[rf.decision_dates(full_px, "monthly")]
    base_e = base_targets.apply(rf.eff, axis=1).reindex(credit_md.index)
    data_audit.update({
        "finrl_repo": "AI4Finance-Foundation/FinRL-Trading",
        "finrl_commit": FINRL_COMMIT,
        "finrl_engine": "src/backtest/backtest_engine.py::BacktestEngine",
        "transaction_cost": TC,
        "integer_positions": False,
        "soft_rebalance_band": rf.BAND,
        "credit_rule": "one-day-lagged HY OAS > trailing 756-trading-day 90th percentile; cap old-money exposure at 1.0x",
        "credit_stress_monthly_decisions_full_warmup": int(credit_md["credit_stress"].fillna(False).sum()),
        "credit_would_cap_monthly_full_warmup": int((credit_md["credit_stress"].fillna(False) & (base_e > 1.0)).sum()),
        "v9_rebalance_count": int(len(base_tr)),
        "v9_1_rebalance_count": int(len(cred_tr)),
        "dca_month_count": int(len(base_acc.index.to_period("M").unique())),
        "total_invested": float(INIT + MONTHLY * len(base_acc.index.to_period("M").unique())),
        "note": "Official FinRL-X BacktestEngine validates unitized old-money execution. Periodic-contribution DCA sleeve is simulated by the frozen V9 cash-flow runner because BacktestEngine exposes initial_capital but no recurring-contribution API.",
    })
    (OUT / "data_audit.json").write_text(json.dumps(data_audit, indent=2))

    print("=== DATA AUDIT ===")
    print(json.dumps(data_audit, indent=2))
    print("=== OFFICIAL FINRL-X UNITIZED ===")
    print(finrl_df.to_string(index=False))
    print("=== REAL-DATA DCA ===")
    print(dca_df.to_string(index=False))


if __name__ == "__main__":
    main()

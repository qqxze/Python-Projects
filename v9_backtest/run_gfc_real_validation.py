#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

import run_frequency_backtest as rf
import run_v10_research_experiments as r1
import run_v10_round2 as r2
import run_finrl_real_validation as real
import run_upside_capture_research as up

OUT = Path("gfc_real_validation")
OUT.mkdir(exist_ok=True)
START = "2007-06-01"
END = "2025-12-31"
WARMUP = "2005-01-03"
INIT = real.INIT
MONTHLY = real.MONTHLY
TC = real.TC


def load_longer_real():
    old_warmup, old_start = real.WARMUP_START, real.REAL_START
    real.WARMUP_START = WARMUP
    real.REAL_START = START
    try:
        tickers = ["QQQ", "SPY", "QLD", "SSO", "TQQQ", "BIL", "SHY", "RSP", "IWM"]
        s = {t: real.download_one(t) for t in tickers}
    finally:
        real.WARMUP_START = old_warmup
        real.REAL_START = old_start

    raw = pd.concat(s.values(), axis=1).sort_index().loc[WARMUP:END]
    full = pd.DataFrame(index=raw.index)
    full["QQQ"] = raw["QQQ"]
    full["SPY"] = raw["SPY"]
    full["QLD"] = raw["QLD"]
    full["SSO"] = raw["SSO"]
    full["TQQQ"] = raw["TQQQ"]
    # Warm-up only: SHY fills dates before BIL inception. From BIL inception onward
    # use BIL, and all execution dates start after BIL exists.
    full["CASH"] = raw["BIL"].combine_first(raw["SHY"])

    common = raw[["QQQ", "SPY", "QLD", "SSO", "SHY"]].dropna().index
    full = full.reindex(common)
    audit = {
        "source": "Yahoo Finance adjusted close via yfinance",
        "period": [START, END],
        "warmup_start": WARMUP,
        "no_synthetic_leveraged_etf_execution": True,
        "QLD_first": str(s["QLD"].first_valid_index().date()),
        "SSO_first": str(s["SSO"].first_valid_index().date()),
        "BIL_first": str(s["BIL"].first_valid_index().date()),
        "cash_rule": "SHY only before BIL inception for feature warm-up; BIL used on all backtest execution dates",
        "TQQQ_unused": True,
    }
    proxies = raw[["RSP", "IWM"]].reindex(full.index).ffill()
    return full, proxies, audit


def add_proxies(meta, proxies):
    m = meta.copy()
    rsp, iwm = proxies["RSP"], proxies["IWM"]
    m["rsp"] = rsp
    m["iwm"] = iwm
    m["rsp_ma210"] = rsp.rolling(210, min_periods=210).mean()
    m["iwm_ma210"] = iwm.rolling(210, min_periods=210).mean()
    m["rsp_mom126"] = rsp / rsp.shift(126) - 1
    m["iwm_mom126"] = iwm / iwm.shift(126) - 1
    m["breadth_healthy"] = (
        (rsp > m["rsp_ma210"]) &
        (iwm > m["iwm_ma210"]) &
        (m["rsp_mom126"] > 0) &
        (m["iwm_mom126"] > 0)
    ).fillna(False)
    m["g_signal"] = (m["growth_confirm"] & ~m["credit_stress"]).fillna(False)
    m["gb_signal"] = (m["g_signal"] & m["breadth_healthy"]).fillna(False)
    return m


def main():
    full_px, proxies, audit = load_longer_real()

    old_rf_start, old_rf_end = rf.START, rf.END
    old_real_start, old_real_end = real.REAL_START, real.REAL_END
    rf.START, rf.END = START, END
    real.REAL_START, real.REAL_END = START, END
    try:
        base_targets, base_meta = rf.strategy(full_px)
        credit = r1.load_credit(Path("data/hy_oas_full.csv"), full_px.index)
        meta = r1.add_research_features(full_px, base_meta, credit)
        meta = add_proxies(meta, proxies)
        v91_targets, v91_meta = r2.make_targets_round2(
            base_targets, meta, bull2=False, credit_gate=True, growth=False
        )
        g_targets, g_meta = up.tilt_targets(v91_targets, v91_meta, "g_signal")
        gb_targets, gb_meta = up.tilt_targets(v91_targets, v91_meta, "gb_signal")

        v91_acc, v91_tr, _, _ = up.simulate_variant(full_px, v91_targets, v91_meta, None)
        g_acc, g_tr, _, _ = up.simulate_variant(full_px, g_targets, g_meta, "g_signal")
        gb_acc, gb_tr, _, _ = up.simulate_variant(full_px, gb_targets, gb_meta, "gb_signal")
        qqq_acc = rf.run_qqq(full_px)

        blend20 = up.make_old_money_blend(v91_acc, full_px["QQQ"], .20)
        blend30 = up.make_old_money_blend(v91_acc, full_px["QQQ"], .30)

        ep = pd.DataFrame(index=full_px.loc[START:END].index)
        for a in ["QQQ", "SPY", "QLD", "SSO"]:
            ep[a] = full_px.loc[START:END, a]
        # Execution cash is strictly BIL on this period.
        old_warmup = real.WARMUP_START
        real.WARMUP_START = WARMUP
        try:
            bil = real.download_one("BIL").reindex(ep.index).ffill()
        finally:
            real.WARMUP_START = old_warmup
        ep["BIL"] = bil
        ep = ep.dropna()
        if ep.index.min() > pd.Timestamp(START) + pd.Timedelta(days=5):
            raise RuntimeError(f"BIL execution history starts too late: {ep.index.min()}")

        p91 = real.build_drift_weight_path(ep, full_px, v91_targets, v91_tr)
        pg = real.build_drift_weight_path(ep, full_px, g_targets, g_tr)
        pgb = real.build_drift_weight_path(ep, full_px, gb_targets, gb_tr)

        _, nav91 = real.run_finrl_engine("V9_1_GFC", ep, p91)
        _, navg = real.run_finrl_engine("G_GFC", ep, pg)
        _, navgb = real.run_finrl_engine("GB_GFC", ep, pgb)

        q = ep.loc[nav91.index, "QQQ"]
        navq = INIT * q / q.iloc[0]
        navb20 = .80 * nav91 + .20 * navq
        navb30 = .70 * nav91 + .30 * navq
    finally:
        rf.START, rf.END = old_rf_start, old_rf_end
        real.REAL_START, real.REAL_END = old_real_start, old_real_end

    unit = pd.DataFrame([
        {"variant": n, **real.metrics_from_values(v)}
        for n, v in [
            ("V9.1", nav91), ("G_GrowthTilt", navg),
            ("GB_BreadthConfirmedGrowth", navgb),
            ("Blend20_OldMoney", navb20), ("Blend30_OldMoney", navb30),
            ("QQQ_BuyHold", navq),
        ]
    ])
    unit.to_csv(OUT/"finrl_unitized_summary.csv", index=False)

    dca_cases = [
        ("V9.1", v91_acc), ("G_GrowthTilt", g_acc),
        ("GB_BreadthConfirmedGrowth", gb_acc),
        ("Blend20_OldMoney", blend20), ("Blend30_OldMoney", blend30),
        ("QQQ_DCA", qqq_acc),
    ]
    dca = pd.DataFrame([{"variant":n, **up.account_perf(a)} for n,a in dca_cases])
    dca["total_invested"] = INIT + MONTHLY * len(v91_acc.index.to_period("M").unique())
    base = dca.set_index("variant").loc["V9.1"]
    for c in ["final","xirr","cagr","max_drawdown","sharpe","calmar"]:
        dca[f"{c}_delta_vs_v91"] = dca[c] - base[c]
    dca["final_pct_vs_v91"] = dca["final"]/base["final"] - 1
    dca.to_csv(OUT/"dca_summary.csv", index=False)

    # Explicit GFC window from market peak through trough.
    gfc_rows = []
    for n,a in dca_cases:
        r = a["return"].loc["2007-10-09":"2009-03-09"]
        if len(r):
            m = r1.unit_metrics(r)
            gfc_rows.append({"variant":n, **m})
    gfc = pd.DataFrame(gfc_rows)
    gfc.to_csv(OUT/"gfc_window.csv", index=False)

    rolling = {}
    for n,a in dca_cases[:-1]:
        rolling[n] = {
            "5y_vs_v91": r1.rolling_compare(a["return"], v91_acc["return"], 5),
            "10y_vs_v91": r1.rolling_compare(a["return"], v91_acc["return"], 10),
            "5y_vs_qqq": r1.rolling_compare(a["return"], qqq_acc["return"], 5),
            "10y_vs_qqq": r1.rolling_compare(a["return"], qqq_acc["return"], 10),
        }
    (OUT/"rolling.json").write_text(json.dumps(rolling, indent=2))

    md = rf.decision_dates(full_px, "monthly")
    dm = meta.reindex(md)
    audit.update({
        "finrl_repo": "AI4Finance-Foundation/FinRL-Trading",
        "finrl_commit": real.FINRL_COMMIT,
        "growth_signal_months": int(dm["g_signal"].fillna(False).sum()),
        "gb_signal_months": int(dm["gb_signal"].fillna(False).sum()),
        "breadth_healthy_months": int(dm["breadth_healthy"].fillna(False).sum()),
        "v91_rebalances": int(len(v91_tr)),
        "G_rebalances": int(len(g_tr)),
        "GB_rebalances": int(len(gb_tr)),
        "rules_unchanged_from_2010_test": True,
        "warning": "This extends the same frozen candidate rules into GFC. SHY is used only for pre-BIL feature warm-up; all executed cash positions in the backtest period use real BIL prices."
    })
    (OUT/"audit.json").write_text(json.dumps(audit, indent=2))

    for n,a in dca_cases:
        a.to_csv(OUT/f"account_{n}.csv")
    pd.DataFrame({
        "V9.1":nav91,"G":navg,"GB":navgb,
        "Blend20":navb20,"Blend30":navb30,"QQQ":navq
    }).to_csv(OUT/"finrl_nav.csv")

    print("=== 2007-2025 OFFICIAL FINRL-X ===")
    print(unit.to_string(index=False))
    print("=== 2007-2025 DCA ===")
    print(dca.to_string(index=False))
    print("=== GFC WINDOW ===")
    print(gfc.to_string(index=False))
    print("=== AUDIT ===")
    print(json.dumps(audit, indent=2))
    print("=== ROLLING ===")
    print(json.dumps(rolling, indent=2))


if __name__ == "__main__":
    main()

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
import run_gfc_real_validation as gfc

OUT = Path("flow_only_growth_ablation")
OUT.mkdir(exist_ok=True)


def prepare_2010():
    full_px, audit = real.load_real_prices()
    credit = r1.load_credit(Path("data/hy_oas_full.csv"), full_px.index)
    base_targets, base_meta = rf.strategy(full_px)
    meta = r1.add_research_features(full_px, base_meta, credit)
    meta, breadth_audit = up.add_breadth_proxy(full_px, meta)
    audit.update(breadth_audit)
    return full_px, base_targets, meta, audit


def prepare_2007():
    full_px, proxies, audit = gfc.load_longer_real()
    credit = r1.load_credit(Path("data/hy_oas_full.csv"), full_px.index)
    base_targets, base_meta = rf.strategy(full_px)
    meta = r1.add_research_features(full_px, base_meta, credit)
    meta = gfc.add_proxies(meta, proxies)
    return full_px, base_targets, meta, audit


def run_period(label, start, end, prepared):
    full_px, base_targets, meta, audit = prepared

    old_s, old_e = rf.START, rf.END
    rf.START, rf.END = start, end
    try:
        v91_targets, v91_meta = r2.make_targets_round2(
            base_targets, meta, bull2=False, credit_gate=True, growth=False
        )
        full_g_targets, full_g_meta = up.tilt_targets(v91_targets, v91_meta, "g_signal")
        full_gb_targets, full_gb_meta = up.tilt_targets(v91_targets, v91_meta, "gb_signal")

        base_acc, base_tr, base_fl, _ = up.simulate_variant(
            full_px, v91_targets, v91_meta, None
        )
        # Attribution variants: old-money target is IDENTICAL to V9.1.
        flow_g_acc, flow_g_tr, flow_g_fl, _ = up.simulate_variant(
            full_px, v91_targets, v91_meta, "g_signal"
        )
        flow_gb_acc, flow_gb_tr, flow_gb_fl, _ = up.simulate_variant(
            full_px, v91_targets, v91_meta, "gb_signal"
        )
        full_g_acc, _, full_g_fl, _ = up.simulate_variant(
            full_px, full_g_targets, full_g_meta, "g_signal"
        )
        full_gb_acc, _, full_gb_fl, _ = up.simulate_variant(
            full_px, full_gb_targets, full_gb_meta, "gb_signal"
        )
        qqq = rf.run_qqq(full_px)
    finally:
        rf.START, rf.END = old_s, old_e

    # Hard attribution check: flow-only variants may not alter existing-capital path.
    max_old_diff_g = float((flow_g_acc["old_value"] - base_acc["old_value"]).abs().max())
    max_old_diff_gb = float((flow_gb_acc["old_value"] - base_acc["old_value"]).abs().max())
    if max_old_diff_g > 1e-6 or max_old_diff_gb > 1e-6:
        raise AssertionError((max_old_diff_g, max_old_diff_gb))

    cases = [
        ("V9.1", base_acc),
        ("FlowOnly_G", flow_g_acc),
        ("FlowOnly_GB", flow_gb_acc),
        ("Full_G", full_g_acc),
        ("Full_GB", full_gb_acc),
        ("QQQ_DCA", qqq),
    ]
    summary = pd.DataFrame([{"variant":n, **up.account_perf(a)} for n,a in cases])
    base = summary.set_index("variant").loc["V9.1"]
    for c in ["final","xirr","cagr","max_drawdown","sharpe","calmar"]:
        summary[f"{c}_delta_vs_v91"] = summary[c] - base[c]
    summary["final_pct_vs_v91"] = summary["final"]/base["final"] - 1
    summary["old_final"] = [float(a["old_value"].iloc[-1]) for _,a in cases]
    summary["flow_final"] = [float(a["flow_value"].iloc[-1]) for _,a in cases]
    summary["total_invested"] = real.INIT + real.MONTHLY * len(base_acc.index.to_period("M").unique())
    summary.to_csv(OUT/f"summary_{label}.csv", index=False)

    rolling = {}
    for n,a in cases[:-1]:
        rolling[n] = {
            "5y_vs_v91": r1.rolling_compare(a["return"], base_acc["return"], 5),
            "10y_vs_v91": r1.rolling_compare(a["return"], base_acc["return"], 10),
        }
    (OUT/f"rolling_{label}.json").write_text(json.dumps(rolling, indent=2))

    windows = {
        "GFC": ("2007-10-09","2009-03-09"),
        "COVID": ("2020-02-19","2020-03-23"),
        "2022": ("2021-11-19","2022-10-14"),
    }
    wr = []
    for wn,(s,e) in windows.items():
        for n,a in cases:
            rr = a["return"].loc[s:e]
            if len(rr) > 2:
                wr.append({"window":wn,"variant":n,**r1.unit_metrics(rr)})
    pd.DataFrame(wr).to_csv(OUT/f"windows_{label}.csv", index=False)

    flow_stats = {}
    for n,fl in [
        ("V9.1",base_fl),("FlowOnly_G",flow_g_fl),("FlowOnly_GB",flow_gb_fl),
        ("Full_G",full_g_fl),("Full_GB",full_gb_fl)
    ]:
        flow_stats[n] = {
            "months": int(len(fl)),
            "mean_new_equity_ratio": float(fl["new_equity_ratio"].mean()),
            "mean_new_qqq_share": float(fl["new_qqq_share"].mean()),
            "months_qqq_share_ge_0_8": int((fl["new_qqq_share"] >= .799999).sum()),
        }
    diagnostics = {
        "period":[start,end],
        "max_existing_capital_path_diff_flowG_vs_v91":max_old_diff_g,
        "max_existing_capital_path_diff_flowGB_vs_v91":max_old_diff_gb,
        "flow_stats":flow_stats,
        "rules":{
            "FlowOnly_G":"Old-money weights exactly V9.1. Only monthly contribution QQQ share rises to 80% in RiskOn/Strong or 70% Neutral when growth_confirm and credit normal.",
            "FlowOnly_GB":"Same, additionally requires RSP/IWM breadth proxy healthy.",
            "Full_G":"Same signal as FlowOnly_G, but also old-money QQQ share 75%/65%.",
            "Full_GB":"Same signal as FlowOnly_GB, but also old-money QQQ share 75%/65%."
        },
        "audit": audit,
        "warning":"Pure attribution/ablation. No thresholds changed after observing prior results."
    }
    (OUT/f"diagnostics_{label}.json").write_text(json.dumps(diagnostics, indent=2))

    for n,a in cases:
        a.to_csv(OUT/f"account_{label}_{n}.csv")

    print(f"=== {label} SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"=== {label} DIAGNOSTICS ===")
    print(json.dumps(diagnostics, indent=2))
    return summary


def main():
    # 2010 all-real baseline used in prior FinRL validation.
    s2010 = run_period("2010_2025", real.REAL_START, real.REAL_END, prepare_2010())

    # GFC-inclusive real execution period.
    old_warm, old_rs, old_re = real.WARMUP_START, real.REAL_START, real.REAL_END
    real.WARMUP_START = gfc.WARMUP
    real.REAL_START = gfc.START
    real.REAL_END = gfc.END
    try:
        s2007 = run_period("2007_2025", gfc.START, gfc.END, prepare_2007())
    finally:
        real.WARMUP_START, real.REAL_START, real.REAL_END = old_warm, old_rs, old_re

    comparison = []
    for label,df in [("2010_2025",s2010),("2007_2025",s2007)]:
        for _,r in df.iterrows():
            comparison.append({
                "period":label,
                "variant":r["variant"],
                "final":r["final"],
                "xirr":r["xirr"],
                "max_drawdown":r["max_drawdown"],
                "sharpe":r["sharpe"],
                "final_pct_vs_v91":r["final_pct_vs_v91"],
                "old_final":r["old_final"],
                "flow_final":r["flow_final"],
            })
    pd.DataFrame(comparison).to_csv(OUT/"comparison.csv", index=False)


if __name__ == "__main__":
    main()

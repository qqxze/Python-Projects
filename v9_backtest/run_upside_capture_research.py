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

OUT = Path("upside_capture_real")
OUT.mkdir(exist_ok=True)
START = real.REAL_START
END = real.REAL_END
INIT = real.INIT
MONTHLY = real.MONTHLY
TC = real.TC


def add_breadth_proxy(full_px: pd.DataFrame, meta: pd.DataFrame):
    rsp = real.download_one("RSP").reindex(full_px.index).ffill()
    iwm = real.download_one("IWM").reindex(full_px.index).ffill()
    m = meta.copy()
    m["rsp"] = rsp
    m["iwm"] = iwm
    m["rsp_ma210"] = rsp.rolling(210, min_periods=210).mean()
    m["iwm_ma210"] = iwm.rolling(210, min_periods=210).mean()
    m["rsp_mom126"] = rsp / rsp.shift(126) - 1.0
    m["iwm_mom126"] = iwm / iwm.shift(126) - 1.0
    m["breadth_healthy"] = (
        (rsp > m["rsp_ma210"]) &
        (iwm > m["iwm_ma210"]) &
        (m["rsp_mom126"] > 0) &
        (m["iwm_mom126"] > 0)
    ).fillna(False)
    # G: existing V9 growth confirmation, but only when credit is not stressed.
    m["g_signal"] = (m["growth_confirm"] & ~m["credit_stress"]).fillna(False)
    # GB: same growth signal, additionally requiring broad participation.
    m["gb_signal"] = (m["g_signal"] & m["breadth_healthy"]).fillna(False)
    audit = {
        "RSP_first": str(rsp.first_valid_index().date()),
        "RSP_last": str(rsp.last_valid_index().date()),
        "RSP_rows": int(rsp.notna().sum()),
        "IWM_first": str(iwm.first_valid_index().date()),
        "IWM_last": str(iwm.last_valid_index().date()),
        "IWM_rows": int(iwm.notna().sum()),
        "breadth_definition": "RSP>210DMA, IWM>210DMA, and both 6M momentum >0",
        "breadth_is_proxy_not_constituent_breadth": True,
    }
    return m, audit


def tilt_targets(v91_targets: pd.DataFrame, meta: pd.DataFrame, signal_col: str):
    e = v91_targets.apply(rf.eff, axis=1)
    qshare = meta["qqq_share"].copy()
    sig = meta[signal_col].fillna(False)
    qshare.loc[sig & meta["regime"].isin(["RiskOn", "StrongRiskOn"])] = 0.75
    qshare.loc[sig & meta["regime"].eq("Neutral")] = 0.65
    rows = []
    for d in v91_targets.index:
        rows.append(
            r1.target_from_exposure(
                float(e.loc[d]),
                float(qshare.loc[d]),
                bool(meta.loc[d, "rs126"] > 0),
            ).values
        )
    out = pd.DataFrame(rows, index=v91_targets.index, columns=rf.ASSETS)
    dm = meta.copy()
    dm["research_qshare"] = qshare
    dm["tilt_applied"] = sig
    return out, dm


def make_flow(signal_col: str | None):
    base = rf.flow_target

    def fn(row):
        t, eq, qs, rec = base(row)
        if signal_col and bool(row.get(signal_col, False)):
            regime = str(row.get("regime", "Neutral"))
            if regime in ("RiskOn", "StrongRiskOn"):
                new_q = 0.80
            elif regime == "Neutral":
                new_q = 0.70
            else:
                new_q = qs
            if new_q != qs:
                t = t.copy()
                t["QQQ"] = eq * new_q
                t["SPY"] = eq * (1 - new_q)
                t["CASH"] = 1 - eq
                qs = new_q
        return t, eq, qs, rec

    return fn


def simulate_variant(full_px, targets, meta, signal_col: str | None):
    old = rf.flow_target
    rf.flow_target = make_flow(signal_col)
    try:
        return rf.simulate(full_px, targets, meta, "monthly")
    finally:
        rf.flow_target = old


def account_perf(o: pd.DataFrame):
    # Same account-return convention as frozen V9: external contributions are
    # removed from that day's return, so drawdown/Sharpe are not distorted.
    r = o["return"].astype(float)
    nav = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.2425
    cagr = float(nav.iloc[-1] ** (1 / yrs) - 1)
    dd = nav / nav.cummax() - 1
    sd = float(r.std(ddof=1))
    return {
        "cagr": cagr,
        "max_drawdown": float(dd.min()),
        "sharpe": float(r.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan,
        "calmar": float(cagr / abs(dd.min())) if dd.min() < 0 else np.nan,
        "final": float(o["value"].iloc[-1]),
        "xirr": float(rf.xirr(o)),
    }


def make_old_money_blend(v91_acc: pd.DataFrame, qqq_prices: pd.Series, beta: float):
    dates = v91_acc.index
    q = qqq_prices.reindex(dates).ffill().astype(float)
    # Only initial existing capital gets the structural QQQ core. New monthly
    # money remains 100% governed by V9.1's dynamic flow policy.
    qcore = INIT * (1 - TC) * q / q.iloc[0]
    dyn_old = v91_acc["old_value"].astype(float)
    flow = v91_acc["flow_value"].astype(float)
    value = (1 - beta) * dyn_old + beta * qcore + flow

    out = v91_acc.copy()
    out["old_value"] = (1 - beta) * dyn_old + beta * qcore
    out["flow_value"] = flow
    out["value"] = value
    prev = value.shift(1)
    out["return"] = ((value - out["contribution"]) / prev - 1).fillna(0.0)
    # Exact mixed-sleeve cost attribution is not used for selection; NAV already
    # contains V9.1 costs and the QQQ core initial 10bp purchase cost.
    out["cost"] = np.nan
    return out


def build_finrl_path(engine_prices, full_px, targets, trades):
    return real.build_drift_weight_path(engine_prices, full_px, targets, trades)


def run_finrl(name, engine_prices, path):
    # Patched by entrypoint so this resolves to pinned official FinRL-X.
    return real.run_finrl_engine(name, engine_prices, path)


def main():
    full_px, audit = real.load_real_prices()

    old_start, old_end = rf.START, rf.END
    rf.START, rf.END = START, END
    try:
        base_targets, base_meta = rf.strategy(full_px)
        credit = r1.load_credit(Path("data/hy_oas_full.csv"), full_px.index)
        meta = r1.add_research_features(full_px, base_meta, credit)
        meta, breadth_audit = add_breadth_proxy(full_px, meta)

        # Frozen V9.1: only credit veto is added to V9.
        v91_targets, v91_meta = r2.make_targets_round2(
            base_targets, meta, bull2=False, credit_gate=True, growth=False
        )

        g_targets, g_meta = tilt_targets(v91_targets, v91_meta, "g_signal")
        gb_targets, gb_meta = tilt_targets(v91_targets, v91_meta, "gb_signal")

        v91_acc, v91_tr, v91_fl, v91_diag = simulate_variant(
            full_px, v91_targets, v91_meta, None
        )
        g_acc, g_tr, g_fl, g_diag = simulate_variant(
            full_px, g_targets, g_meta, "g_signal"
        )
        gb_acc, gb_tr, gb_fl, gb_diag = simulate_variant(
            full_px, gb_targets, gb_meta, "gb_signal"
        )
        qqq_acc = rf.run_qqq(full_px)
    finally:
        rf.START, rf.END = old_start, old_end

    # Existing-capital structural beta blends. Contributions remain V9.1 dynamic.
    blend20 = make_old_money_blend(v91_acc, full_px["QQQ"], 0.20)
    blend30 = make_old_money_blend(v91_acc, full_px["QQQ"], 0.30)

    # Official FinRL-X old-money validation for V9.1, G and GB.
    ep = pd.DataFrame(index=full_px.loc[START:END].index)
    ep["QQQ"] = full_px.loc[START:END, "QQQ"]
    ep["SPY"] = full_px.loc[START:END, "SPY"]
    ep["QLD"] = full_px.loc[START:END, "QLD"]
    ep["SSO"] = full_px.loc[START:END, "SSO"]
    ep["BIL"] = full_px.loc[START:END, "CASH"]
    ep = ep.dropna()

    p_v91 = build_finrl_path(ep, full_px, v91_targets, v91_tr)
    p_g = build_finrl_path(ep, full_px, g_targets, g_tr)
    p_gb = build_finrl_path(ep, full_px, gb_targets, gb_tr)

    _, nav_v91 = run_finrl("V9_1", ep, p_v91)
    _, nav_g = run_finrl("V9_1_Growth", ep, p_g)
    _, nav_gb = run_finrl("V9_1_Growth_Breadth", ep, p_gb)

    q = ep.loc[nav_v91.index, "QQQ"]
    nav_qqq = INIT * q / q.iloc[0]
    nav_blend20 = 0.80 * nav_v91 + 0.20 * nav_qqq
    nav_blend30 = 0.70 * nav_v91 + 0.30 * nav_qqq

    finrl_rows = []
    for name, nav in [
        ("V9.1", nav_v91),
        ("G_GrowthTilt", nav_g),
        ("GB_BreadthConfirmedGrowth", nav_gb),
        ("Blend20_OldMoney", nav_blend20),
        ("Blend30_OldMoney", nav_blend30),
        ("QQQ_BuyHold", nav_qqq),
    ]:
        finrl_rows.append({"variant": name, **real.metrics_from_values(nav)})
    finrl = pd.DataFrame(finrl_rows)
    finrl.to_csv(OUT / "finrl_unitized_summary.csv", index=False)

    dca_rows = []
    for name, acc in [
        ("V9.1", v91_acc),
        ("G_GrowthTilt", g_acc),
        ("GB_BreadthConfirmedGrowth", gb_acc),
        ("Blend20_OldMoney", blend20),
        ("Blend30_OldMoney", blend30),
        ("QQQ_DCA", qqq_acc),
    ]:
        dca_rows.append({"variant": name, **account_perf(acc)})
        acc.to_csv(OUT / f"account_{name}.csv")
    dca = pd.DataFrame(dca_rows)
    dca["total_invested"] = INIT + MONTHLY * len(v91_acc.index.to_period("M").unique())
    base = dca.set_index("variant").loc["V9.1"]
    for c in ["final", "xirr", "cagr", "max_drawdown", "sharpe", "calmar"]:
        dca[f"{c}_delta_vs_v91"] = dca[c] - base[c]
    dca["final_pct_vs_v91"] = dca["final"] / base["final"] - 1
    dca.to_csv(OUT / "dca_summary.csv", index=False)

    # Rolling account-return comparisons are contribution-neutral.
    cases = {
        "V9.1": v91_acc,
        "G_GrowthTilt": g_acc,
        "GB_BreadthConfirmedGrowth": gb_acc,
        "Blend20_OldMoney": blend20,
        "Blend30_OldMoney": blend30,
    }
    rolling = {}
    for name, acc in cases.items():
        rolling[name] = {
            "5y_vs_v91": r1.rolling_compare(acc["return"], v91_acc["return"], 5),
            "10y_vs_v91": r1.rolling_compare(acc["return"], v91_acc["return"], 10),
            "5y_vs_qqq": r1.rolling_compare(acc["return"], qqq_acc["return"], 5),
            "10y_vs_qqq": r1.rolling_compare(acc["return"], qqq_acc["return"], 10),
        }
    (OUT / "rolling.json").write_text(json.dumps(rolling, indent=2))

    md = rf.decision_dates(full_px, "monthly")
    dm = meta.reindex(md)
    audit.update(breadth_audit)
    audit.update({
        "period": [START, END],
        "finrl_repo": "AI4Finance-Foundation/FinRL-Trading",
        "finrl_commit": real.FINRL_COMMIT,
        "rules_frozen_before_run": {
            "G": "V9.1 exposure unchanged. Existing V9 growth_confirm AND credit normal. Old QQQ share=75% RiskOn/Strong, 65% Neutral; new-money QQQ share=80%/70%.",
            "GB": "G rule plus breadth proxy healthy: RSP and IWM above 210DMA and both 126d momentum >0.",
            "Blend20_30": "Only existing capital: 20% or 30% buy-and-hold QQQ core plus the remaining V9.1 old-money sleeve. Monthly contributions remain fully dynamic V9.1 flow; no inter-sleeve rebalancing.",
        },
        "growth_signal_months": int(dm["g_signal"].fillna(False).sum()),
        "gb_signal_months": int(dm["gb_signal"].fillna(False).sum()),
        "breadth_healthy_months": int(dm["breadth_healthy"].fillna(False).sum()),
        "V91_rebalances": int(len(v91_tr)),
        "G_rebalances": int(len(g_tr)),
        "GB_rebalances": int(len(gb_tr)),
        "warning": "RSP/IWM are investable breadth proxies, not constituent-level breadth. This is research on an already-developed historical sample, not true OOS.",
    })
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2))

    v91_tr.to_csv(OUT / "trades_v91.csv", index=False)
    g_tr.to_csv(OUT / "trades_g.csv", index=False)
    gb_tr.to_csv(OUT / "trades_gb.csv", index=False)
    pd.DataFrame({
        "V9.1": nav_v91,
        "G_GrowthTilt": nav_g,
        "GB_BreadthConfirmedGrowth": nav_gb,
        "Blend20_OldMoney": nav_blend20,
        "Blend30_OldMoney": nav_blend30,
        "QQQ": nav_qqq,
    }).to_csv(OUT / "finrl_nav.csv")

    print("=== OFFICIAL FINRL-X OLD-MONEY ===")
    print(finrl.to_string(index=False))
    print("=== DCA / FULL ACCOUNT ===")
    print(dca.to_string(index=False))
    print("=== AUDIT ===")
    print(json.dumps(audit, indent=2))
    print("=== ROLLING ===")
    print(json.dumps(rolling, indent=2))


if __name__ == "__main__":
    main()

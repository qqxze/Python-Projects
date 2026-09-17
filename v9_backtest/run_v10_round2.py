#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

import run_frequency_backtest as rf
from run_formal_backtest import load_prices, crisis_stats
import run_v10_research_experiments as r1

OUT=Path('v10_round2_results'); OUT.mkdir(exist_ok=True)
END='2025-12-31'; ASSETS=rf.ASSETS


def target_from_exposure(exposure,qshare,rs_positive):
    return r1.target_from_exposure(exposure,qshare,rs_positive)


def make_targets_round2(base_targets,meta,bull2,credit_gate,growth):
    out=base_targets.copy()
    base_e=base_targets.apply(rf.eff,axis=1)
    e=base_e.copy(); qshare=meta['qqq_share'].copy()

    bull_mask=pd.Series(False,index=meta.index)
    if bull2:
        # No new leverage parameter: in persistent bulls, if raw V9 e8 is already >1,
        # remove only the baseline 15% shrink-to-1 caused by defense_strength=0.85.
        bull_mask=(meta['bull_persist'] & meta['regime'].isin(['RiskOn','StrongRiskOn']) & (meta['e8']>1.0)).fillna(False)
        raw=meta['e8'].clip(upper=1.34)
        e=e.where(~bull_mask,np.maximum(e,raw))

    growth_mask=pd.Series(False,index=meta.index)
    if growth:
        growth_mask=meta['growth_confirm'].fillna(False)
        qshare=qshare.copy()
        qshare.loc[growth_mask & meta['regime'].isin(['RiskOn','StrongRiskOn'])]=.75
        qshare.loc[growth_mask & meta['regime'].eq('Neutral')]=.65

    credit_mask=pd.Series(False,index=meta.index)
    if credit_gate:
        credit_mask=meta['credit_stress'].fillna(False)
        e=e.where(~credit_mask,np.minimum(e,1.0))

    arr=[]
    for d in out.index:
        arr.append(target_from_exposure(e.loc[d],qshare.loc[d],bool(meta.loc[d,'rs126']>0)).values)
    out=pd.DataFrame(np.asarray(arr),index=out.index,columns=ASSETS)

    dm=meta.copy(); dm['research_exposure']=e; dm['research_qshare']=qshare
    dm['bull2_applied']=bull_mask & (e>base_e+1e-12)
    dm['credit_applied']=credit_mask & (e<=1.0) & (base_e>1.0)
    dm['growth_applied']=growth_mask
    return out,dm


def make_flow_func(use_growth):
    return r1.make_flow_func(use_growth)


def run_variant(px,base_targets,meta,bull2,credit_gate,growth):
    targets,dm=make_targets_round2(base_targets,meta,bull2,credit_gate,growth)
    old=rf.flow_target; rf.flow_target=make_flow_func(growth)
    try:o,tr,fl,dg=rf.simulate(px,targets,dm,'monthly')
    finally:rf.flow_target=old
    return o,tr,fl,dg,dm


def unit_metrics(r): return r1.unit_metrics(r)
def rolling_compare(a,b,years): return r1.rolling_compare(a,b,years)
def era_table(cases): return r1.era_table(cases)


def main():
    px=load_prices(Path('data')).loc[:END]
    base_targets,base_meta=rf.strategy(px)
    credit=r1.load_credit(Path('data/hy_oas_full.csv'),px.index)
    meta=r1.add_research_features(px,base_meta,credit)

    specs=[
      ('Baseline',False,False,False),('A2_BullPersistence',True,False,False),('B_CreditGate',False,True,False),('C_GrowthTilt',False,False,True),
      ('A2B_Bull_Credit',True,True,False),('A2C_Bull_Growth',True,False,True),('BC_Credit_Growth',False,True,True),('A2BC_All',True,True,True)
    ]
    qqq=rf.run_qqq(px); cases={}; detail={}; rows=[]
    for name,a,b,c in specs:
        o,tr,fl,dg,dm=run_variant(px,base_targets,meta,a,b,c); cases[name]=o; detail[name]=(tr,fl,dm)
        rows.append({'variant':name,'bull2':a,'credit':b,'growth':c,**rf.perf(o,dg)})
        o.to_csv(OUT/f'account_{name}.csv'); tr.to_csv(OUT/f'trades_{name}.csv',index=False); fl.to_csv(OUT/f'flow_{name}.csv')
    rows.append({'variant':'QQQ','bull2':False,'credit':False,'growth':False,**rf.perf(qqq,{'decision_count':0,'rebalance_count':0,'annual_turnover_notional_to_avg_nav':0,'total_cost':float(qqq.cost.sum())})})
    summary=pd.DataFrame(rows); base=summary.set_index('variant').loc['Baseline']
    for col in ['final','xirr','cagr','max_drawdown','sharpe','calmar']:summary[f'{col}_delta_vs_base']=summary[col]-base[col]
    summary['final_pct_vs_base']=summary['final']/base['final']-1; summary.to_csv(OUT/'summary.csv',index=False)

    rolling={}
    for name,o in cases.items():
        rolling[name]={
          '5y_vs_qqq':rolling_compare(o['return'],qqq['return'],5),'10y_vs_qqq':rolling_compare(o['return'],qqq['return'],10),
          '5y_vs_baseline':rolling_compare(o['return'],cases['Baseline']['return'],5),'10y_vs_baseline':rolling_compare(o['return'],cases['Baseline']['return'],10)}
    (OUT/'rolling.json').write_text(json.dumps(rolling,indent=2))
    era_table({**cases,'QQQ':qqq}).to_csv(OUT/'eras.csv',index=False)
    (OUT/'crisis.json').write_text(json.dumps({n:crisis_stats(o['return']) for n,o in {**cases,'QQQ':qqq}.items()},indent=2))

    # Decision-date diagnostics, not just daily-condition counts.
    md=rf.decision_dates(px,'monthly'); dm=meta.loc[md]
    base_e=base_targets.apply(rf.eff,axis=1).loc[md]
    diagnostics={
      'credit_non_null_obs':int(meta['hy_oas'].notna().sum()),'credit_first':str(meta['hy_oas'].first_valid_index().date()) if meta['hy_oas'].first_valid_index() is not None else None,
      'credit_last':str(meta['hy_oas'].last_valid_index().date()) if meta['hy_oas'].last_valid_index() is not None else None,
      'credit_stress_days':int(meta['credit_stress'].sum()),'credit_stress_monthly_decisions':int(dm['credit_stress'].sum()),
      'credit_would_cap_monthly':int((dm['credit_stress'] & (base_e>1.0)).sum()),
      'bull_persist_days':int(meta['bull_persist'].sum()),'bull2_monthly_candidates':int((dm['bull_persist'] & dm['regime'].isin(['RiskOn','StrongRiskOn']) & (dm['e8']>1.0)).sum()),
      'growth_confirm_days':int(meta['growth_confirm'].sum()),
      'rules':{
        'A2':'Persistent bull + RiskOn/Strong + raw e8>1: use raw e8 rather than 1+0.85*(e8-1); max remains 1.34x. No new leverage/floor parameter.',
        'B':'One-day-lagged HY OAS > trailing 756-trading-day 90th percentile: cap old-money exposure at 1.0x only.',
        'C':'Unchanged from Round1 Growth Tilt.'},
      'warning':'Round2 was specified after learning Round1 A was a no-op and FRED history was truncated. Treat as research, not OOS.'}
    (OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    print('=== SUMMARY ===');print(summary.to_string(index=False));print('=== DIAGNOSTICS ===');print(json.dumps(diagnostics,indent=2));print('=== ROLLING ===');print(json.dumps(rolling,indent=2))

if __name__=='__main__':main()

#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json
import itertools
import numpy as np
import pandas as pd

import run_frequency_backtest as rf
from run_formal_backtest import load_prices, crisis_stats

OUT=Path('v10_research_results'); OUT.mkdir(exist_ok=True)
START='2000-01-03'; END='2025-12-31'
ASSETS=rf.ASSETS


def load_credit(path: Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    x=pd.read_csv(path)
    date_col=x.columns[0]; val_col=x.columns[1]
    x[date_col]=pd.to_datetime(x[date_col], errors='coerce')
    x[val_col]=pd.to_numeric(x[val_col], errors='coerce')
    s=x.dropna(subset=[date_col]).set_index(date_col)[val_col].sort_index()
    s=s.reindex(index).ffill()
    q90=s.rolling(756,min_periods=504).quantile(.90)
    # One-day publication/availability lag before use in target construction.
    stress=(s.shift(1)>q90.shift(1)).fillna(False)
    return pd.DataFrame({'hy_oas':s,'hy_oas_q90_3y':q90,'credit_stress':stress},index=index)


def add_research_features(px: pd.DataFrame, meta: pd.DataFrame, credit: pd.DataFrame) -> pd.DataFrame:
    m=meta.copy()
    q=px['QQQ']; s=px['SPY']; ratio=q/s
    m['q']=q; m['spy']=s
    m['rs63_extra']=ratio/ratio.shift(63)-1.0
    above=(q>m['ma210']).astype(float)
    m['above210_share_252']=above.rolling(252,min_periods=210).mean()
    m['bull_persist']=(
        (m['above210_share_252']>=.80) &
        (q>m['ma210']) & (s>m['spy_ma210']) &
        (m['ma210_slope']>0) & (m['mom126']>0)
    ).fillna(False)
    m=m.join(credit,how='left')
    m['credit_stress']=m['credit_stress'].fillna(False).astype(bool)
    m['growth_confirm']=(m['bull_persist'] & (m['rs63_extra']>0) & (m['rs126']>0)).fillna(False)
    return m


def target_from_exposure(exposure: float, qshare: float, rs_positive: bool) -> pd.Series:
    e=float(np.clip(exposure,0,1.34)); qs=float(np.clip(qshare,0,1))
    w=pd.Series(0.0,index=ASSETS)
    if e<=1.0:
        w['QQQ']=e*qs; w['SPY']=e*(1-qs); w['CASH']=1-e
    else:
        lev=e-1.0
        unlev=1.0-lev
        w['QQQ']=unlev*qs; w['SPY']=unlev*(1-qs)
        w['QLD' if rs_positive else 'SSO']=lev
    return w


def make_targets(base_targets: pd.DataFrame, meta: pd.DataFrame, bull: bool, credit_gate: bool, growth: bool) -> tuple[pd.DataFrame,pd.DataFrame]:
    out=base_targets.copy()
    diag=meta.copy()
    base_e=base_targets.apply(rf.eff,axis=1)
    qshare=meta['qqq_share'].copy()
    e=base_e.copy()

    bull_mask=pd.Series(False,index=meta.index)
    if bull:
        bull_mask=(meta['bull_persist'] & meta['regime'].isin(['Neutral','RiskOn','StrongRiskOn'])).fillna(False)
        e=e.where(~bull_mask,np.maximum(e,1.0))

    growth_mask=pd.Series(False,index=meta.index)
    if growth:
        growth_mask=meta['growth_confirm'].fillna(False)
        # Only boost tech tilt in confirmed persistent bulls; total exposure unchanged.
        qshare=qshare.copy()
        qshare.loc[growth_mask & meta['regime'].isin(['RiskOn','StrongRiskOn'])]=.75
        qshare.loc[growth_mask & meta['regime'].eq('Neutral')]=.65

    credit_mask=pd.Series(False,index=meta.index)
    if credit_gate:
        credit_mask=meta['credit_stress'].fillna(False)
        # Credit only vetoes leverage. It never forces equity below 1x.
        e=e.where(~credit_mask,np.minimum(e,1.0))

    for d in out.index:
        out.loc[d]=target_from_exposure(e.loc[d],qshare.loc[d],bool(meta.loc[d,'rs126']>0))

    diag['research_exposure']=e
    diag['research_qshare']=qshare
    diag['bull_floor_applied']=bull_mask
    diag['credit_cap_applied']=credit_mask & (base_e>1.0)
    diag['growth_tilt_applied']=growth_mask
    return out,diag


def make_flow_func(use_growth: bool):
    base_func=rf.flow_target
    def flow_func(row):
        t,eq,qs,rec=base_func(row)
        if use_growth and bool(row.get('growth_confirm',False)):
            regime=str(row.get('regime','Neutral'))
            if regime in ('RiskOn','StrongRiskOn'):
                new_q=.80
            elif regime=='Neutral':
                new_q=.70
            else:
                new_q=qs
            if new_q!=qs:
                t=t.copy(); t['QQQ']=eq*new_q; t['SPY']=eq*(1-new_q); t['CASH']=1-eq; qs=new_q
        return t,eq,qs,rec
    return flow_func


def run_variant(px,base_targets,meta,name,bull,credit_gate,growth):
    targets,diagmeta=make_targets(base_targets,meta,bull,credit_gate,growth)
    old_flow=rf.flow_target
    rf.flow_target=make_flow_func(growth)
    try:
        o,tr,fl,dg=rf.simulate(px,targets,diagmeta,'monthly')
    finally:
        rf.flow_target=old_flow
    return o,tr,fl,dg,diagmeta


def unit_metrics(r: pd.Series):
    r=r.dropna(); nav=(1+r).cumprod(); yrs=(r.index[-1]-r.index[0]).days/365.2425
    dd=nav/nav.cummax()-1; sd=float(r.std(ddof=1))
    return {'total_return':float(nav.iloc[-1]-1),'cagr':float(nav.iloc[-1]**(1/yrs)-1),'mdd':float(dd.min()),'sharpe':float(r.mean()/sd*np.sqrt(252)) if sd>0 else np.nan}


def rolling_compare(a: pd.Series,b: pd.Series,years:int):
    rows=[]
    starts=pd.date_range(max(a.index.min(),b.index.min()).normalize(),min(a.index.max(),b.index.max()).normalize(),freq='MS')
    for s in starts:
        e=s+pd.DateOffset(years=years)-pd.Timedelta(days=1)
        ar=a.loc[s:e]; br=b.loc[s:e]
        if len(ar)<int(252*years*.8) or len(br)<int(252*years*.8): continue
        am=unit_metrics(ar); bm=unit_metrics(br)
        rows.append(((1+am['total_return'])/(1+bm['total_return'])-1,am['mdd']>bm['mdd']))
    if not rows:return {'n':0}
    x=np.array([r[0] for r in rows]); low=np.array([r[1] for r in rows],dtype=float)
    return {'n':len(rows),'beat_wealth_pct':float((x>0).mean()),'median_wealth_excess':float(np.median(x)),'p10_wealth_excess':float(np.quantile(x,.10)),'p90_wealth_excess':float(np.quantile(x,.90)),'lower_mdd_pct':float(low.mean())}


def era_table(cases):
    eras=[('dotcom','2000-01-03','2002-10-09'),('gfc','2007-10-09','2009-03-09'),('post_gfc','2009-03-10','2020-02-19'),('covid_crash','2020-02-19','2020-03-23'),('bear_2022','2021-11-19','2022-10-14'),('ai_bull','2022-10-15','2025-12-31')]
    rows=[]
    for name,o in cases.items():
        for era,s,e in eras:
            m=unit_metrics(o.loc[s:e,'return'])
            rows.append({'variant':name,'era':era,**m})
    return pd.DataFrame(rows)


def main():
    px=load_prices(Path('data')).loc[:END]
    base_targets,base_meta=rf.strategy(px)
    credit=load_credit(Path('data/hy_oas.csv'),px.index)
    meta=add_research_features(px,base_meta,credit)

    specs=[
        ('Baseline',False,False,False),
        ('A_BullPersistence',True,False,False),
        ('B_CreditGate',False,True,False),
        ('C_GrowthTilt',False,False,True),
        ('AB_Bull_Credit',True,True,False),
        ('AC_Bull_Growth',True,False,True),
        ('BC_Credit_Growth',False,True,True),
        ('ABC_All',True,True,True),
    ]
    qqq=rf.run_qqq(px)
    cases={}; logs={}; rows=[]
    for name,a,b,c in specs:
        o,tr,fl,dg,dm=run_variant(px,base_targets,meta,name,a,b,c)
        cases[name]=o; logs[name]=(tr,fl,dm)
        p=rf.perf(o,dg)
        rows.append({'variant':name,'bull':a,'credit':b,'growth':c,**p})
        o.to_csv(OUT/f'account_{name}.csv'); tr.to_csv(OUT/f'trades_{name}.csv',index=False); fl.to_csv(OUT/f'flow_{name}.csv')
    qperf=rf.perf(qqq,{'decision_count':0,'rebalance_count':0,'annual_turnover_notional_to_avg_nav':0,'total_cost':float(qqq.cost.sum())})
    rows.append({'variant':'QQQ','bull':False,'credit':False,'growth':False,**qperf})
    summary=pd.DataFrame(rows)
    base=summary.set_index('variant').loc['Baseline']
    for col in ['final','xirr','cagr','max_drawdown','sharpe','calmar']:
        summary[f'{col}_delta_vs_base']=summary[col]-base[col]
    summary['final_pct_vs_base']=summary['final']/base['final']-1
    summary.to_csv(OUT/'summary.csv',index=False)

    rolling={}
    for name,o in cases.items():
        rolling[name]={
            '5y_vs_qqq':rolling_compare(o['return'],qqq['return'],5),
            '10y_vs_qqq':rolling_compare(o['return'],qqq['return'],10),
            '5y_vs_baseline':rolling_compare(o['return'],cases['Baseline']['return'],5),
            '10y_vs_baseline':rolling_compare(o['return'],cases['Baseline']['return'],10),
        }
    (OUT/'rolling.json').write_text(json.dumps(rolling,indent=2))
    eras=era_table({**cases,'QQQ':qqq}); eras.to_csv(OUT/'eras.csv',index=False)
    crisis={name:crisis_stats(o['return']) for name,o in {**cases,'QQQ':qqq}.items()}; (OUT/'crisis.json').write_text(json.dumps(crisis,indent=2))

    diagnostics={
        'bull_persist_trading_days':int(meta['bull_persist'].sum()),
        'credit_stress_trading_days':int(meta['credit_stress'].sum()),
        'growth_confirm_trading_days':int(meta['growth_confirm'].sum()),
        'monthly_decisions':int(len(rf.decision_dates(px,'monthly'))),
        'rules':{
            'A_BullPersistence':'252d share above 210DMA >=80%, QQQ and SPY >210DMA, 210DMA slope>0, 6m momentum>0; in Neutral/RiskOn/StrongRiskOn floor old-money effective exposure at 1.0x.',
            'B_CreditGate':'ICE BofA US HY OAS (FRED BAMLH0A0HYM2), one-day lag; if OAS > trailing 756-trading-day 90th percentile, cap old-money effective exposure at 1.0x; never reduce below 1x by itself.',
            'C_GrowthTilt':'Bull Persistence + QQQ/SPY 63d and 126d relative strength both >0; old-money QQQ share 75% in RiskOn/Strong and 65% Neutral, new-money 80%/70%; total equity exposure unchanged.'
        },
        'warning':'This is an A/B research experiment on the already-developed V9 historical sample, not true out-of-sample validation. No parameters are chosen after observing these experiment results.'
    }
    (OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    print('=== SUMMARY ==='); print(summary.to_string(index=False))
    print('=== DIAGNOSTICS ==='); print(json.dumps(diagnostics,indent=2))
    print('=== ROLLING ==='); print(json.dumps(rolling,indent=2))
    print('=== ERAS ==='); print(eras.to_string(index=False))

if __name__=='__main__': main()

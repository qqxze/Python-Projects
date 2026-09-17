#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

import run_frequency_backtest as rf
from run_formal_backtest import load_prices

OUT = Path('robustness_results')
OUT.mkdir(exist_ok=True)
BASE_START = '2000-01-03'
BASE_END = '2025-12-31'
BASE_TC = 0.001
INIT = 1_000_000.0
MONTHLY = 20_000.0
RNG_SEED = 20260917


def run_case(px, targets, meta, start, end, tc=BASE_TC, delay=1):
    old_start, old_end, old_tc, old_make = rf.START, rf.END, rf.TC, rf.make_schedule
    rf.START, rf.END, rf.TC = start, end, tc

    def delayed_schedule(px0, targets0, frequency):
        rows = []
        idx = px0.index
        for d in rf.decision_dates(px0, frequency):
            pos = idx.searchsorted(pd.Timestamp(d), side='right')
            ex_pos = pos + max(0, int(delay) - 1)
            if ex_pos < len(idx):
                rows.append((idx[ex_pos], targets0.loc[d]))
        s = pd.DataFrame({d:r for d,r in rows}).T if rows else pd.DataFrame(columns=rf.ASSETS)
        if len(s):
            s.columns = rf.ASSETS
            s = s[~s.index.duplicated(keep='last')].sort_index()
        return s

    rf.make_schedule = delayed_schedule
    try:
        out, tr, fl, diag = rf.simulate(px, targets, meta, 'monthly')
        qqq = rf.run_qqq(px)
        return out, tr, fl, diag, qqq
    finally:
        rf.START, rf.END, rf.TC, rf.make_schedule = old_start, old_end, old_tc, old_make


def metrics(o):
    r = o['return'].dropna()
    nav = (1+r).cumprod()
    years = (r.index[-1]-r.index[0]).days/365.2425
    cagr = float(nav.iloc[-1]**(1/years)-1)
    dd = nav/nav.cummax()-1
    sd = float(r.std(ddof=1))
    return {
        'final': float(o['value'].iloc[-1]),
        'xirr': float(rf.xirr(o)),
        'cagr': cagr,
        'max_drawdown': float(dd.min()),
        'sharpe': float(r.mean()/sd*np.sqrt(252)) if sd>0 else np.nan,
        'calmar': float(cagr/abs(dd.min())) if dd.min()<0 else np.nan,
    }


def unit_metrics(r):
    r = r.dropna()
    if len(r) < 2:
        return {'total_return': np.nan, 'cagr': np.nan, 'max_drawdown': np.nan, 'sharpe': np.nan}
    nav = (1+r).cumprod()
    years = (r.index[-1]-r.index[0]).days/365.2425
    cagr = float(nav.iloc[-1]**(1/years)-1) if years > 0 else np.nan
    dd = nav/nav.cummax()-1
    sd = float(r.std(ddof=1))
    return {
        'total_return': float(nav.iloc[-1]-1),
        'cagr': cagr,
        'max_drawdown': float(dd.min()),
        'sharpe': float(r.mean()/sd*np.sqrt(252)) if sd>0 else np.nan,
    }


def starting_years(px, targets, meta):
    rows=[]
    for year in [2000,2003,2006,2009,2012,2015,2018,2021]:
        start=f'{year}-01-03'
        if pd.Timestamp(start) > px.index.max():
            continue
        o,tr,fl,dg,q=run_case(px,targets,meta,start,BASE_END,BASE_TC,1)
        vm, qm = metrics(o), metrics(q)
        rows.append({
            'start_year':year,
            'v9_final':vm['final'],'v9_xirr':vm['xirr'],'v9_cagr':vm['cagr'],'v9_mdd':vm['max_drawdown'],'v9_sharpe':vm['sharpe'],
            'qqq_final':qm['final'],'qqq_xirr':qm['xirr'],'qqq_cagr':qm['cagr'],'qqq_mdd':qm['max_drawdown'],'qqq_sharpe':qm['sharpe'],
            'v9_vs_qqq_final_pct': vm['final']/qm['final']-1,
            'rebalance_count':int(dg['rebalance_count']),
            'annual_turnover':float(dg['annual_turnover_notional_to_avg_nav']),
        })
    return pd.DataFrame(rows)


def cost_sensitivity(px, targets, meta):
    rows=[]
    for tc in [0.0005,0.0010,0.0020]:
        o,tr,fl,dg,q=run_case(px,targets,meta,BASE_START,BASE_END,tc,1)
        vm=metrics(o)
        rows.append({'cost_bp':int(round(tc*10000)),**vm,'rebalance_count':int(dg['rebalance_count']),'annual_turnover':float(dg['annual_turnover_notional_to_avg_nav']),'total_cost':float(dg['total_cost'])})
    return pd.DataFrame(rows)


def delay_sensitivity(px, targets, meta):
    rows=[]
    for delay in [1,3,5]:
        o,tr,fl,dg,q=run_case(px,targets,meta,BASE_START,BASE_END,BASE_TC,delay)
        vm=metrics(o)
        rows.append({'execution_delay_trading_days':delay,**vm,'rebalance_count':int(dg['rebalance_count']),'annual_turnover':float(dg['annual_turnover_notional_to_avg_nav']),'total_cost':float(dg['total_cost'])})
    return pd.DataFrame(rows)


def rolling_windows(base, qqq, years):
    idx = base.index
    starts = pd.date_range(idx.min().normalize(), idx.max().normalize(), freq='MS')
    rows=[]
    for s in starts:
        e=s+pd.DateOffset(years=years)-pd.Timedelta(days=1)
        br=base.loc[(base.index>=s)&(base.index<=e),'return']
        qr=qqq.loc[(qqq.index>=s)&(qqq.index<=e),'return']
        if len(br)<int(252*years*0.8) or len(qr)<int(252*years*0.8):
            continue
        bm=unit_metrics(br); qm=unit_metrics(qr)
        rows.append({
            'start':str(s.date()),'end':str(e.date()),
            'v9_total_return':bm['total_return'],'v9_cagr':bm['cagr'],'v9_mdd':bm['max_drawdown'],'v9_sharpe':bm['sharpe'],
            'qqq_total_return':qm['total_return'],'qqq_cagr':qm['cagr'],'qqq_mdd':qm['max_drawdown'],'qqq_sharpe':qm['sharpe'],
            'v9_beat_wealth':bool(bm['total_return']>qm['total_return']),
            'v9_lower_mdd':bool(bm['max_drawdown']>qm['max_drawdown']),
            'wealth_excess':float((1+bm['total_return'])/(1+qm['total_return'])-1),
        })
    df=pd.DataFrame(rows)
    summary={
        'years':years,'n':int(len(df)),
        'beat_wealth_pct':float(df.v9_beat_wealth.mean()) if len(df) else np.nan,
        'lower_mdd_pct':float(df.v9_lower_mdd.mean()) if len(df) else np.nan,
        'median_wealth_excess':float(df.wealth_excess.median()) if len(df) else np.nan,
        'p10_wealth_excess':float(df.wealth_excess.quantile(.10)) if len(df) else np.nan,
        'p90_wealth_excess':float(df.wealth_excess.quantile(.90)) if len(df) else np.nan,
        'median_v9_cagr':float(df.v9_cagr.median()) if len(df) else np.nan,
        'median_v9_mdd':float(df.v9_mdd.median()) if len(df) else np.nan,
    }
    return df, summary


def era_table(base, qqq):
    eras=[
        ('dotcom','2000-01-03','2002-10-09'),
        ('recovery_2003_2007','2003-01-01','2007-10-09'),
        ('gfc','2007-10-09','2009-03-09'),
        ('post_gfc_expansion','2009-03-10','2020-02-19'),
        ('covid_crash','2020-02-19','2020-03-23'),
        ('covid_bull','2020-03-24','2021-11-19'),
        ('bear_2022','2021-11-19','2022-10-14'),
        ('ai_bull_2023_2025','2022-10-15','2025-12-31'),
    ]
    rows=[]
    for name,s,e in eras:
        vm=unit_metrics(base.loc[s:e,'return']); qm=unit_metrics(qqq.loc[s:e,'return'])
        rows.append({'era':name,'start':s,'end':e,
                     'v9_total_return':vm['total_return'],'v9_cagr':vm['cagr'],'v9_mdd':vm['max_drawdown'],'v9_sharpe':vm['sharpe'],
                     'qqq_total_return':qm['total_return'],'qqq_cagr':qm['cagr'],'qqq_mdd':qm['max_drawdown'],'qqq_sharpe':qm['sharpe']})
    return pd.DataFrame(rows)


def pseudo_walkforward(px, targets, meta):
    folds=[('2000_2004','2000-01-03','2004-12-31'),('2005_2009','2005-01-03','2009-12-31'),('2010_2014','2010-01-04','2014-12-31'),('2015_2019','2015-01-02','2019-12-31'),('2020_2025','2020-01-02','2025-12-31')]
    rows=[]
    for name,s,e in folds:
        o,tr,fl,dg,q=run_case(px,targets,meta,s,e,BASE_TC,1)
        vm,qm=metrics(o),metrics(q)
        rows.append({'fold':name,'start':s,'end':e,
                     'v9_final':vm['final'],'v9_xirr':vm['xirr'],'v9_cagr':vm['cagr'],'v9_mdd':vm['max_drawdown'],'v9_sharpe':vm['sharpe'],
                     'qqq_final':qm['final'],'qqq_xirr':qm['xirr'],'qqq_cagr':qm['cagr'],'qqq_mdd':qm['max_drawdown'],'qqq_sharpe':qm['sharpe'],
                     'v9_vs_qqq_final_pct':vm['final']/qm['final']-1})
    return pd.DataFrame(rows)


def monthly_returns(df):
    per=df.index.to_period('M')
    return (1+df['return']).groupby(per).prod()-1


def irr_monthly(returns):
    # Beginning-of-month contribution; same convention for both strategies.
    bal=INIT
    for r in returns:
        bal += MONTHLY
        bal *= 1+r
    n=len(returns)
    def f(mr):
        if mr <= -0.999999: return 1e100
        pv=-INIT
        for k in range(n):
            pv -= MONTHLY/((1+mr)**k)
        pv += bal/((1+mr)**n)
        return pv
    lo,hi=-.99,1.0
    flo,fhi=f(lo),f(hi)
    for _ in range(100):
        if np.sign(flo)!=np.sign(fhi): break
        hi*=2; fhi=f(hi)
    for _ in range(100):
        mid=(lo+hi)/2; fm=f(mid)
        if np.sign(fm)==np.sign(flo): lo=mid; flo=fm
        else: hi=mid
    mr=(lo+hi)/2
    return bal,(1+mr)**12-1


def bootstrap(base, qqq, n_paths=2000, block=12):
    b=monthly_returns(base); q=monthly_returns(qqq)
    ix=b.index.intersection(q.index); b=b.loc[ix].to_numpy(); q=q.loc[ix].to_numpy(); n=len(ix)
    rng=np.random.default_rng(RNG_SEED)
    starts=np.arange(0,max(1,n-block+1))
    rows=[]
    for p in range(n_paths):
        take=[]
        while len(take)<n:
            s=int(rng.choice(starts)); take.extend(range(s,min(s+block,n)))
        take=np.array(take[:n],dtype=int)
        br=b[take]; qr=q[take]
        bf,birr=irr_monthly(br); qf,qirr=irr_monthly(qr)
        bnav=np.cumprod(1+br); qnav=np.cumprod(1+qr)
        bmdd=float(np.min(bnav/np.maximum.accumulate(bnav)-1)); qmdd=float(np.min(qnav/np.maximum.accumulate(qnav)-1))
        rows.append((p,bf,qf,birr,qirr,bmdd,qmdd,bf/qf-1))
    df=pd.DataFrame(rows,columns=['path','v9_final','qqq_final','v9_irr','qqq_irr','v9_mdd','qqq_mdd','wealth_excess'])
    s={
        'n_paths':n_paths,'block_months':block,'seed':RNG_SEED,
        'v9_final_median':float(df.v9_final.median()),'v9_final_p10':float(df.v9_final.quantile(.10)),'v9_final_p90':float(df.v9_final.quantile(.90)),
        'qqq_final_median':float(df.qqq_final.median()),
        'v9_beat_qqq_wealth_pct':float((df.v9_final>df.qqq_final).mean()),
        'v9_lower_mdd_pct':float((df.v9_mdd>df.qqq_mdd).mean()),
        'median_wealth_excess':float(df.wealth_excess.median()),
        'v9_irr_median':float(df.v9_irr.median()),'v9_irr_p10':float(df.v9_irr.quantile(.10)),'v9_irr_p90':float(df.v9_irr.quantile(.90)),
        'v9_mdd_median':float(df.v9_mdd.median()),'v9_mdd_p10':float(df.v9_mdd.quantile(.10)),'v9_mdd_p90':float(df.v9_mdd.quantile(.90)),
        'note':'Paired 12-month moving-block bootstrap of realized monthly V9 and QQQ return paths. This preserves paired market co-movement and local serial structure, but does not recompute the state-dependent policy on synthetic histories.'
    }
    return df,s


def main():
    px=load_prices(Path('data')).loc[:BASE_END]
    targets,meta=rf.strategy(px)
    base,tr,fl,diag,qqq=run_case(px,targets,meta,BASE_START,BASE_END,BASE_TC,1)
    baseline={'v9':metrics(base),'qqq':metrics(qqq),'diag':diag}

    sy=starting_years(px,targets,meta); sy.to_csv(OUT/'starting_years.csv',index=False)
    cs=cost_sensitivity(px,targets,meta); cs.to_csv(OUT/'cost_sensitivity.csv',index=False)
    ds=delay_sensitivity(px,targets,meta); ds.to_csv(OUT/'delay_sensitivity.csv',index=False)
    r5,s5=rolling_windows(base,qqq,5); r5.to_csv(OUT/'rolling_5y.csv',index=False)
    r10,s10=rolling_windows(base,qqq,10); r10.to_csv(OUT/'rolling_10y.csv',index=False)
    eras=era_table(base,qqq); eras.to_csv(OUT/'eras.csv',index=False)
    wf=pseudo_walkforward(px,targets,meta); wf.to_csv(OUT/'pseudo_walkforward.csv',index=False)
    boot,boots=bootstrap(base,qqq,2000,12); boot.to_csv(OUT/'bootstrap_paths.csv',index=False)
    base.to_csv(OUT/'baseline_account.csv'); qqq.to_csv(OUT/'qqq_account.csv')

    report={
        'spec':{
            'strategy':'Frozen V9 Production candidate: Monthly strategic regime, max 1.34x, no emergency brake, 5pp soft rebalance, original dynamic new-money engine',
            'period':[BASE_START,BASE_END],'initial':INIT,'monthly_contribution':MONTHLY,'base_transaction_cost':BASE_TC,
            'tests':['starting years','rolling 5y','rolling 10y','eras','cost 5/10/20bp','execution delay 1/3/5 trading days','paired 12m block bootstrap','pseudo walk-forward'],
            'warning':'Pseudo walk-forward is not true out-of-sample because V9 was designed with knowledge of this historical sample. No parameters are refit inside folds.'
        },
        'baseline':baseline,'rolling_5y_summary':s5,'rolling_10y_summary':s10,'bootstrap_summary':boots
    }
    (OUT/'summary.json').write_text(json.dumps(report,indent=2))
    print('=== BASELINE ==='); print(json.dumps(baseline,indent=2))
    print('=== STARTING YEARS ==='); print(sy.to_string(index=False))
    print('=== COST ==='); print(cs.to_string(index=False))
    print('=== DELAY ==='); print(ds.to_string(index=False))
    print('=== ROLLING ==='); print(json.dumps({'5y':s5,'10y':s10},indent=2))
    print('=== ERAS ==='); print(eras.to_string(index=False))
    print('=== PSEUDO WALK-FORWARD ==='); print(wf.to_string(index=False))
    print('=== BOOTSTRAP ==='); print(json.dumps(boots,indent=2))

if __name__=='__main__': main()

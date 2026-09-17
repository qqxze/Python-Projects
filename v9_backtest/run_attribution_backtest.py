#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.strategies.v9_final_strategy import V9FinalStrategy, V9FinalConfig
from run_formal_backtest import load_prices, crisis_stats

ASSETS=['QQQ','SPY','QLD','SSO','TQQQ','CASH']
START='2000-01-03'; END='2025-12-31'; MONTHLY=20_000.0
DRAG_ANNUAL=.0095; BAND=.05; TC=.001


def v9(px):
    cfg=replace(V9FinalConfig(),tqqq_capital_cap=0.,strong_trend_boost=0.,final_exposure_cap=1.34,strong_band=(1.15,1.40))
    st=V9FinalStrategy(cfg)
    targets,meta=st.daily_targets(px)
    sig,audit=st.generate_signals(px)
    return st,targets,meta,sig.reindex(columns=ASSETS),audit

def xirr(flows):
    ds=[pd.Timestamp(d) for d,_ in flows]; vals=np.array([v for _,v in flows],float)
    tt=np.array([(d-ds[0]).days/365.2425 for d in ds])
    def f(r): return np.sum(vals/(1+r)**tt)
    lo,hi=-.999,10.; fl,fh=f(lo),f(hi)
    for _ in range(200):
        if np.sign(fl)!=np.sign(fh): break
        hi*=2; fh=f(hi)
    if np.sign(fl)==np.sign(fh): return np.nan
    for _ in range(200):
        m=(lo+hi)/2; fm=f(m)
        if np.sign(fm)==np.sign(fl): lo=m; fl=fm
        else: hi=m
    return (lo+hi)/2

def perf(ret,value,contrib):
    r=ret.dropna(); nav=(1+r).cumprod(); yrs=(r.index[-1]-r.index[0]).days/365.2425
    cagr=float(nav.iloc[-1]**(1/yrs)-1); dd=nav/nav.cummax()-1; mdd=float(dd.min()); sd=float(r.std(ddof=1))
    flows=[(d,-float(x)) for d,x in contrib[contrib>0].items()]+[(value.index[-1],float(value.iloc[-1]))]
    return {'final':float(value.iloc[-1]),'xirr':float(xirr(flows)),'cagr':cagr,'max_drawdown':mdd,'sharpe':float(r.mean()/sd*np.sqrt(252)) if sd>0 else np.nan,'calmar':cagr/abs(mdd) if mdd<0 else np.nan}

def monthly_state(px,targets,meta):
    # State known at prior month-end and executable from next month's first trading day.
    me=px.groupby(px.index.to_period('M')).tail(1).index
    t=targets.loc[me].copy(); m=meta.loc[me,['exposure','qqq_share','rs126','regime']].copy()
    t.index=t.index.to_period('M'); m.index=m.index.to_period('M')
    return t,m

def monthly_proxy(px,targets,meta):
    t,m=monthly_state(px,targets,meta)
    p=px[['QQQ','SPY','CASH']].groupby(px.index.to_period('M')).last()
    mr=p.pct_change(); periods=[x for x in mr.index if x.start_time>=pd.Timestamp('2000-01-01') and x.end_time<=pd.Timestamp('2025-12-31')]
    bal=0.; prev=0.; rows=[]; flows=[]
    for per in periods:
        prevper=per-1
        if prevper not in m.index or per not in mr.index: continue
        d=px[px.index.to_period('M')==per].index[0]; contrib=MONTHLY; bal+=contrib; flows.append((d,-contrib))
        E=float(m.loc[prevper,'exposure']); qs=float(m.loc[prevper,'qqq_share'])
        rq,rs,rc=float(mr.loc[per,'QQQ']),float(mr.loc[per,'SPY']),float(mr.loc[per,'CASH'])
        req=qs*rq+(1-qs)*rs
        if E<=1: rp=E*req+(1-E)*rc
        else: rp=E*req-(E-1)*rc-(E-1)*DRAG_ANNUAL/12
        old=bal; bal*=1+rp; twr=(bal-contrib)/prev-1 if prev>0 else rp
        rows.append((d,twr,bal,contrib,E,qs)); prev=bal
    out=pd.DataFrame(rows,columns=['date','return','value','contribution','exposure','qqq_share']).set_index('date')
    return out

def daily_proxy_monthly_signal(px,targets,meta):
    _,m=monthly_state(px,targets,meta); dates=px.loc[START:END].index
    rq=px.QQQ.pct_change().fillna(0); rs=px.SPY.pct_change().fillna(0); rc=px.CASH.pct_change().fillna(0)
    bal=0.; prev=0.; last=None; rows=[]
    for d in dates:
        per=d.to_period('M'); prevper=per-1
        if prevper not in m.index: continue
        contrib=0.
        if per!=last: contrib=MONTHLY; bal+=contrib; last=per
        E=float(m.loc[prevper,'exposure']); qs=float(m.loc[prevper,'qqq_share']); req=qs*float(rq.loc[d])+(1-qs)*float(rs.loc[d])
        if E<=1: rp=E*req+(1-E)*float(rc.loc[d])
        else: rp=E*req-(E-1)*float(rc.loc[d])-(E-1)*DRAG_ANNUAL/252
        bal*=1+rp; twr=(bal-contrib)/prev-1 if prev>0 else rp
        rows.append((d,twr,bal,contrib,E,qs)); prev=bal
    return pd.DataFrame(rows,columns=['date','return','value','contribution','exposure','qqq_share']).set_index('date')

def effective(w): return float(w.QQQ+w.SPY+2*w.QLD+2*w.SSO+3*w.TQQQ)
def norm(v):
    s=float(v.sum()); return v/s if s>0 else pd.Series(0.,index=ASSETS)
def mark(v,p,prevp):
    if prevp is None: return v
    rr=(p/prevp).replace([np.inf,-np.inf],np.nan).fillna(1.0); return v*rr

def fill_underweight(v,amount,target,cost_rate):
    target=target.reindex(ASSETS).fillna(0.); target=target/target.sum(); total=float(v.sum())+amount
    desired=total*target; gap=(desired-v).clip(lower=0.)
    if gap.sum()<=0: alloc=amount*target
    else: alloc=amount*gap/gap.sum()
    buy_notional=float(alloc[['QQQ','SPY','QLD','SSO','TQQQ']].sum()); cost=cost_rate*buy_notional
    alloc=alloc.copy();
    if alloc.CASH>=cost: alloc.CASH-=cost
    else:
        scale=max(0.,(amount-cost)/amount); alloc*=scale
    return v+alloc,cost

def rebalance(v,target,cost_rate):
    target=target.reindex(ASSETS).fillna(0.); target=target/target.sum(); total=float(v.sum()); desired=total*target
    notional=float((desired-v).abs().sum()); cost=cost_rate*notional
    return (total-cost)*target,notional,cost

def simulate_holdings(px, schedule, cost_rate=0., soft=False, band=.05):
    dates=px.loc[START:END].index; schedule=schedule.sort_index().reindex(columns=ASSETS)
    active=None; v=pd.Series(0.,index=ASSETS); prevp=None; prev_total=0.; last=None; rows=[]; turns=0.; costs=0.
    # initialize active target from latest schedule state on/before start if possible, otherwise first target
    pre=schedule.loc[:dates[0]]
    active=(pre.iloc[-1] if len(pre) else schedule.iloc[0]).copy()
    for d in dates:
        p=px.loc[d,ASSETS]; v=mark(v,p,prevp); contrib=0.; daycost=0.
        per=d.to_period('M')
        if per!=last:
            contrib=MONTHLY; last=per; v,c=fill_underweight(v,contrib,active,cost_rate); daycost+=c; costs+=c
        if d in schedule.index:
            active=schedule.loc[d].copy(); do=True
            if soft and v.sum()>0:
                ce=effective(norm(v)); te=effective(active); lev=float(norm(v)[['QLD','SSO','TQQQ']].sum())
                do=(abs(ce-te)>=band) or (te<=1.0 and lev>.01)
            if do and v.sum()>0:
                v,n,c=rebalance(v,active,cost_rate); turns+=n; daycost+=c; costs+=c
        total=float(v.sum()); r=(total-contrib)/prev_total-1 if prev_total>0 else 0.
        rows.append((d,r,total,contrib,daycost,effective(norm(v)))); prev_total=total; prevp=p
    out=pd.DataFrame(rows,columns=['date','return','value','contribution','cost','exposure']).set_index('date')
    yrs=(out.index[-1]-out.index[0]).days/365.2425
    return out,{'annual_turnover_notional_over_avg_nav':float(turns/out.value.mean()/yrs) if out.value.mean()>0 else 0.,'total_cost':float(costs)}

def monthly_real_schedule(px,targets):
    me=px.groupby(px.index.to_period('M')).tail(1).index; t=targets.loc[me].copy(); rows=[]
    for i in range(len(t)-1):
        per=t.index[i].to_period('M')+1; idx=px[px.index.to_period('M')==per].index
        if len(idx): rows.append((idx[0],t.iloc[i]))
    return pd.DataFrame({d:r for d,r in rows}).T.reindex(columns=ASSETS)

def main():
    od=Path('attribution_results'); od.mkdir(exist_ok=True)
    px=load_prices(Path('data')).loc[:END]; st,targets,meta,sig,audit=v9(px)
    cases={}
    cases['A_monthly_proxy']=monthly_proxy(px,targets,meta)
    cases['B_daily_proxy_monthly_signal']=daily_proxy_monthly_signal(px,targets,meta)
    ms=monthly_real_schedule(px,targets)
    c,diag=simulate_holdings(px,ms,0.,False); cases['C_real_LETF_monthly_signal_no_cost']=c
    d,diagd=simulate_holdings(px,sig,0.,False); cases['D_real_LETF_weekly_t1_no_cost']=d
    e,diage=simulate_holdings(px,sig,TC,False); cases['E_real_LETF_weekly_t1_10bp']=e
    f,diagf=simulate_holdings(px,sig,TC,True,BAND); cases['F_real_LETF_weekly_t1_soft_10bp']=f
    rows=[]; diagnostics={'C':diag,'D':diagd,'E':diage,'F':diagf}
    for name,o in cases.items():
        p=perf(o['return'],o['value'],o['contribution']); p['variant']=name; rows.append(p); o.to_csv(od/f'{name}.csv')
    summary=pd.DataFrame(rows)[['variant','final','xirr','cagr','max_drawdown','sharpe','calmar']]
    # sequential attribution versus previous row
    summary['delta_final_vs_prev']=summary['final'].diff(); summary['delta_xirr_pp_vs_prev']=summary['xirr'].diff()*100; summary['delta_mdd_pp_vs_prev']=summary['max_drawdown'].diff()*100
    summary.to_csv(od/'summary.csv',index=False)
    (od/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    (od/'method.json').write_text(json.dumps({
      'period':[START,END],'monthly_contribution':MONTHLY,'total_contributions':6240000.0,
      'strategy':'Frozen V9 1.34x, no TQQQ boost',
      'A':'Monthly effective-exposure proxy, prior month-end signal, 0.95% annual leverage drag, no explicit trading cost',
      'B':'Same prior month-end signal and exposure proxy, but daily underlying compounding',
      'C':'Same monthly signal, real/synthetic daily-reset QLD/SSO paths, holdings drift, monthly exact rebalance, no cost',
      'D':'Real/synthetic daily-reset QLD/SSO, frozen weekly t+1 V9 signals, exact rebalance, no cost',
      'E':'D plus 10bp trading cost',
      'F':'E plus 5pp Soft Rebalance and fill-underweight monthly contributions',
      'legacy_reference_only':{'final':64430000.0,'xirr':0.1507,'max_drawdown':-0.399,'strategy_cagr':0.1166},
      'warning':'Legacy screenshot code was not persisted; A is a controlled reconstruction, not byte-for-byte reproduction.'
    },indent=2))
    (od/'crisis_stats.json').write_text(json.dumps({k:crisis_stats(v['return']) for k,v in cases.items()},indent=2))
    print(summary.to_string(index=False)); print(json.dumps(diagnostics,indent=2))

if __name__=='__main__': main()

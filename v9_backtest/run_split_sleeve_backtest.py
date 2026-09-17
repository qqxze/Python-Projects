#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.strategies.v9_final_strategy import V9FinalStrategy,V9FinalConfig
from run_formal_backtest import load_prices, crisis_stats, rolling_summary

ASSETS=['QQQ','SPY','QLD','SSO','TQQQ','CASH']; START='2000-01-03'; END='2025-12-31'
INIT=1_000_000.0; MONTHLY=20_000.0; TC=.001; BAND=.05
NEW_EQ={'Crisis':.50,'RiskOff':.60,'Neutral':.80,'RiskOn':.95,'StrongRiskOn':1.0}

def eff(w): return float(w.QQQ+w.SPY+2*w.QLD+2*w.SSO+3*w.TQQQ)
def weights(v): return v/v.sum() if v.sum()>0 else pd.Series(0.,index=ASSETS)
def rebalance(v,t):
    t=t.reindex(ASSETS).fillna(0.); t=t/t.sum(); total=float(v.sum()); desired=total*t
    notional=float((desired-v).abs().sum()); cost=TC*notional; return (total-cost)*t,notional,cost

def need_rebalance(v,t):
    w=weights(v); ce,te=eff(w),eff(t); lev=float(w.QLD+w.SSO+w.TQQQ)
    if te<=1.0 and lev>.01: return True
    if ce-te>=BAND or te-ce>=BAND: return True
    return False

def alloc_flow(v,amount,regime,qshare,aggressive):
    eqr=1.0 if aggressive else NEW_EQ.get(str(regime),.80); eqamt=amount*eqr
    cur=float(v.QQQ+v.SPY); curq=float(v.QQQ/cur) if cur>0 else qshare
    if curq<qshare-.02: q=eqamt
    elif curq>qshare+.02: q=0.
    else: q=eqamt*qshare
    a=pd.Series(0.,index=ASSETS); a.QQQ=q; a.SPY=eqamt-q; a.CASH=amount-eqamt
    cost=TC*float(a.QQQ+a.SPY)
    if a.CASH>=cost: a.CASH-=cost
    else: a*=max(0.,(a.sum()-cost)/a.sum())
    return v+a,cost

def mark(v,p,prev):
    if prev is None:return v
    rr=(p/prev).replace([np.inf,-np.inf],np.nan).fillna(1.0); return v*rr

def run_split(px,sig,meta,aggressive=False):
    dates=px.loc[START:END].index; sig=sig.reindex(columns=ASSETS); lag=meta.shift(1).reindex(dates).ffill()
    old=pd.Series(0.,index=ASSETS); flow=pd.Series(0.,index=ASSETS); prevp=None; prev_total=0.; last=None
    rows=[]; trades=[]; initialized=False
    for d in dates:
        p=px.loc[d,ASSETS]; old=mark(old,p,prevp); flow=mark(flow,p,prevp)
        if not initialized:
            # Initial existing capital uses the first executable target known on/after start.
            target=sig.loc[:d].iloc[-1] if len(sig.loc[:d]) else sig.iloc[0]
            old=INIT*target/target.sum(); cost=TC*INIT; old*= (INIT-cost)/old.sum(); initialized=True
        contrib=0.; daycost=0.; per=d.to_period('M')
        if per!=last:
            contrib=MONTHLY; last=per
            rg=lag.loc[d,'regime'] if pd.notna(lag.loc[d,'regime']) else 'Neutral'; qs=float(lag.loc[d,'qqq_share']) if pd.notna(lag.loc[d,'qqq_share']) else .5
            flow,c=alloc_flow(flow,contrib,rg,qs,aggressive); daycost+=c
        if d in sig.index and need_rebalance(old,sig.loc[d]):
            old,n,c=rebalance(old,sig.loc[d]); daycost+=c; trades.append((d,n,c,eff(weights(old)),eff(sig.loc[d])))
        total=float(old.sum()+flow.sum())
        r=(total-contrib)/prev_total-1 if prev_total>0 else 0.
        rows.append((r,total,float(old.sum()),float(flow.sum()),contrib,daycost,eff(weights(old))))
        prev_total=total; prevp=p
    out=pd.DataFrame(rows,index=dates,columns=['return','value','old_value','flow_value','contribution','cost','old_exposure'])
    tr=pd.DataFrame(trades,columns=['date','notional','cost','post_exposure','target_exposure']); return out,tr

def run_qqq(px):
    dates=px.loc[START:END].index; old=INIT*(1-TC); flow=0.; prevp=None; prev_total=old; last=None; rows=[]
    for i,d in enumerate(dates):
        p=float(px.loc[d,'QQQ'])
        if prevp is not None: old*=p/prevp; flow*=p/prevp
        contrib=0.; cost=0.; per=d.to_period('M')
        if per!=last:
            contrib=MONTHLY; last=per; cost=TC*contrib; flow+=contrib-cost
        total=old+flow; r=(total-contrib)/prev_total-1 if i>0 else 0.
        rows.append((r,total,old,flow,contrib,cost,1.0)); prev_total=total; prevp=p
    return pd.DataFrame(rows,index=dates,columns=['return','value','old_value','flow_value','contribution','cost','old_exposure'])

def xirr(o):
    flows=[(o.index[0],-INIT)]+[(d,-float(x)) for d,x in o.loc[o.contribution>0,'contribution'].items()]+[(o.index[-1],float(o.value.iloc[-1]))]
    dates=[pd.Timestamp(d) for d,_ in flows]; vals=np.array([v for _,v in flows]); tt=np.array([(d-dates[0]).days/365.2425 for d in dates])
    def f(x):return np.sum(vals/(1+x)**tt)
    lo,hi=-.999,10.; fl,fh=f(lo),f(hi)
    for _ in range(200):
        if np.sign(fl)!=np.sign(fh):break
        hi*=2;fh=f(hi)
    for _ in range(200):
        m=(lo+hi)/2;fm=f(m)
        if np.sign(fm)==np.sign(fl):lo=m;fl=fm
        else:hi=m
    return (lo+hi)/2

def perf(o,tr=None):
    r=o['return']; nav=(1+r).cumprod(); yrs=(r.index[-1]-r.index[0]).days/365.2425; c=float(nav.iloc[-1]**(1/yrs)-1); dd=nav/nav.cummax()-1; m=float(dd.min()); sd=r.std(ddof=1)
    turn=float(tr.notional.sum()/o.value.mean()/yrs) if tr is not None and len(tr) else 0.
    return {'cagr':c,'max_drawdown':m,'sharpe':float(r.mean()/sd*np.sqrt(252)) if sd>0 else np.nan,'calmar':c/abs(m) if m<0 else np.nan,'final':float(o.value.iloc[-1]),'xirr':float(xirr(o)),'old_final':float(o.old_value.iloc[-1]),'flow_final':float(o.flow_value.iloc[-1]),'annual_rebalance_notional_to_avg_nav':turn,'total_cost':float(o.cost.sum()),'max_old_exposure':float(o.old_exposure.max())}

def main():
    od=Path('split_results');od.mkdir(exist_ok=True); px=load_prices(Path('data')).loc[:END]
    cfg=replace(V9FinalConfig(),tqqq_capital_cap=0.,strong_trend_boost=0.,final_exposure_cap=1.34)
    st=V9FinalStrategy(cfg); sig,_=st.generate_signals(px);_,meta=st.daily_targets(px)
    q=run_qqq(px); b,tb=run_split(px,sig,meta,False); a,ta=run_split(px,sig,meta,True)
    cases={'QQQ_all':(q,None),'V9_old_plus_flow_base':(b,tb),'V9_old_plus_flow_100eq':(a,ta)}
    rows=[]
    for k,(o,tr) in cases.items():
        rows.append({'variant':k,**perf(o,tr)});o.to_csv(od/f'account_{k}.csv');
        if tr is not None:tr.to_csv(od/f'trades_{k}.csv',index=False)
    pd.DataFrame(rows).to_csv(od/'summary.csv',index=False)
    (od/'crisis_stats.json').write_text(json.dumps({k:crisis_stats(o['return']) for k,(o,_) in cases.items()},indent=2))
    rolling={k:{'3y_vs_QQQ':rolling_summary(o['return'],q['return'],3),'5y_vs_QQQ':rolling_summary(o['return'],q['return'],5)} for k,(o,_) in cases.items() if k!='QQQ_all'}
    (od/'rolling.json').write_text(json.dumps(rolling,indent=2))
    (od/'diagnostics.json').write_text(json.dumps({'initial':INIT,'monthly':MONTHLY,'tc':TC,'band':BAND,'new_eq':NEW_EQ,'architecture':'separate existing-capital defensive sleeve + no-sell monthly flow sleeve'},indent=2))
    print(pd.DataFrame(rows).to_string(index=False)); print(json.dumps({k:crisis_stats(o['return']) for k,(o,_) in cases.items()},indent=2))
if __name__=='__main__':main()

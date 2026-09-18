#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.strategies.v9_final_strategy import V9FinalStrategy, V9FinalConfig
from run_formal_backtest import load_prices, crisis_stats, rolling_summary

ASSETS=['QQQ','SPY','QLD','SSO','TQQQ','CASH']
START='2000-01-03'; END='2025-12-31'
INIT=1_000_000.0; MONTHLY=20_000.0; TC=.001; BAND=.05
BASE_EQ={'Crisis':.50,'RiskOff':.60,'Neutral':.80,'RiskOn':.95,'StrongRiskOn':1.00}


def eff(w): return float(w.QQQ+w.SPY+2*w.QLD+2*w.SSO+3*w.TQQQ)
def norm(v):
    s=float(v.sum()); return v/s if s>0 else pd.Series(0.,index=ASSETS)
def mark(v,p,prev):
    if prev is None: return v
    rr=(p/prev).replace([np.inf,-np.inf],np.nan).fillna(1.0)
    return v*rr

def dd_bonus(dd):
    if not np.isfinite(dd): return 0.0
    if dd<=-.40: return .40
    if dd<=-.30: return .30
    if dd<=-.20: return .20
    if dd<=-.10: return .10
    return 0.0

def flow_target(meta_row):
    regime=str(meta_row['regime']) if pd.notna(meta_row['regime']) else 'Neutral'
    dd=float(meta_row['drawdown']) if pd.notna(meta_row['drawdown']) else 0.0
    rs=float(meta_row['rs126']) if pd.notna(meta_row['rs126']) else 0.0
    mom63=float(meta_row['mom63']) if pd.notna(meta_row['mom63']) else 0.0
    q=float(meta_row['q']) if pd.notna(meta_row['q']) else np.nan
    ma105=float(meta_row['ma105']) if pd.notna(meta_row['ma105']) else np.nan
    recovery=(dd<=-.20) and (mom63>0) and np.isfinite(q) and np.isfinite(ma105) and (q>ma105)
    eq=min(1.0, BASE_EQ.get(regime,.80) + dd_bonus(dd))
    if recovery and dd<=-.30: qshare=.70
    elif recovery: qshare=.65
    elif regime=='StrongRiskOn': qshare=.70 if rs>0 else .55
    elif regime=='RiskOn': qshare=.65 if rs>0 else .50
    elif regime=='Neutral': qshare=.60 if rs>0 else .50
    else: qshare=.40
    t=pd.Series(0.,index=ASSETS)
    t.QQQ=eq*qshare; t.SPY=eq*(1-qshare); t.CASH=1-eq
    return t,eq,qshare,recovery

def buy_flow(flow,amount,target):
    a=amount*target
    cost=TC*float(a[['QQQ','SPY','QLD','SSO','TQQQ']].sum())
    if a.CASH>=cost: a.CASH-=cost
    else: a*=max(0.,(amount-cost)/amount)
    return flow+a,cost

def need_rebalance(v,target):
    w=norm(v); ce,te=eff(w),eff(target); lev=float(w[['QLD','SSO','TQQQ']].sum())
    return (abs(ce-te)>=BAND) or (te<=1.0 and lev>.01)

def rebalance(v,target):
    target=target.reindex(ASSETS).fillna(0.); target=target/target.sum()
    total=float(v.sum()); desired=total*target
    notional=float((desired-v).abs().sum()); cost=TC*notional
    return (total-cost)*target,notional,cost

def next_trading_day(index,d):
    loc=index.searchsorted(pd.Timestamp(d),side='right')
    return index[loc] if loc<len(index) else None

def decision_dates(px,frequency):
    if frequency=='monthly':
        return px.groupby(px.index.to_period('M')).tail(1).index
    weekly=px.groupby(px.index.to_period('W-FRI')).tail(1).index
    if frequency=='weekly': return weekly
    if frequency=='biweekly': return weekly[1::2]
    raise ValueError(frequency)

def make_schedule(px,targets,frequency):
    rows=[]
    for d in decision_dates(px,frequency):
        ex=next_trading_day(px.index,d)
        if ex is not None: rows.append((ex,targets.loc[d]))
    s=pd.DataFrame({d:r for d,r in rows}).T if rows else pd.DataFrame(columns=ASSETS)
    s.columns=ASSETS
    return s[~s.index.duplicated(keep='last')].sort_index()

def strategy(px):
    cfg=replace(V9FinalConfig(),tqqq_capital_cap=0.,strong_trend_boost=0.,final_exposure_cap=1.34,strong_band=(1.15,1.40))
    st=V9FinalStrategy(cfg); targets,meta=st.daily_targets(px)
    meta=meta.copy(); meta['q']=px['QQQ']
    return targets,meta

def simulate(px,targets,meta,frequency):
    schedule=make_schedule(px,targets,frequency)
    dates=px.loc[START:END].index
    # New-money allocation always uses information known by prior close; unchanged across frequencies.
    known=meta.shift(1).reindex(dates).ffill()
    old=pd.Series(0.,index=ASSETS); flow=pd.Series(0.,index=ASSETS)
    prevp=None; prev_total=0.; last_month=None; initialized=False
    rows=[]; trades=[]; flowlog=[]; total_cost=0.
    active_pre=schedule.loc[:dates[0]]
    active=(active_pre.iloc[-1] if len(active_pre) else schedule.iloc[0]).copy()
    for d in dates:
        p=px.loc[d,ASSETS]; old=mark(old,p,prevp); flow=mark(flow,p,prevp)
        if not initialized:
            old=INIT*active/active.sum(); init_cost=TC*INIT; old*=((INIT-init_cost)/old.sum()); total_cost+=init_cost; initialized=True
        contrib=0.; daycost=0.
        per=d.to_period('M')
        if per!=last_month:
            contrib=MONTHLY; last_month=per
            ft,eq,qs,rec=flow_target(known.loc[d]); flow,c=buy_flow(flow,contrib,ft); daycost+=c; total_cost+=c
            flowlog.append((d,eq,qs,rec,float(known.loc[d,'drawdown']),str(known.loc[d,'regime'])))
        if d in schedule.index:
            active=schedule.loc[d].copy()
            if need_rebalance(old,active):
                old,n,c=rebalance(old,active); trades.append((d,n,c,eff(norm(old)),eff(active))); daycost+=c; total_cost+=c
        total=float(old.sum()+flow.sum())
        r=(total-contrib)/prev_total-1 if prev_total>0 else 0.
        rows.append((d,r,total,float(old.sum()),float(flow.sum()),contrib,daycost,eff(norm(old))))
        prev_total=total; prevp=p
    out=pd.DataFrame(rows,columns=['date','return','value','old_value','flow_value','contribution','cost','old_exposure']).set_index('date')
    tr=pd.DataFrame(trades,columns=['date','notional','cost','post_exposure','target_exposure'])
    fl=pd.DataFrame(flowlog,columns=['date','new_equity_ratio','new_qqq_share','recovery','drawdown','regime']).set_index('date')
    yrs=(out.index[-1]-out.index[0]).days/365.2425
    turnover=float(tr.notional.sum()/out.value.mean()/yrs) if len(tr) else 0.
    return out,tr,fl,{'decision_count':int(len(schedule.loc[START:END])),'rebalance_count':int(len(tr)),'annual_turnover_notional_to_avg_nav':turnover,'total_cost':float(total_cost)}

def run_qqq(px):
    dates=px.loc[START:END].index; old=INIT*(1-TC); flow=0.; prevp=None; prev_total=old; last=None; rows=[]
    for i,d in enumerate(dates):
        p=float(px.loc[d,'QQQ'])
        if prevp is not None: old*=p/prevp; flow*=p/prevp
        contrib=0.; cost=0.; per=d.to_period('M')
        if per!=last:
            contrib=MONTHLY; last=per; cost=TC*contrib; flow+=contrib-cost
        total=old+flow; r=(total-contrib)/prev_total-1 if i>0 else 0.
        rows.append((d,r,total,old,flow,contrib,cost,1.0)); prev_total=total; prevp=p
    return pd.DataFrame(rows,columns=['date','return','value','old_value','flow_value','contribution','cost','old_exposure']).set_index('date')

def xirr(o):
    flows=[(o.index[0],-INIT)]+[(d,-float(x)) for d,x in o.loc[o.contribution>0,'contribution'].items()]+[(o.index[-1],float(o.value.iloc[-1]))]
    ds=[pd.Timestamp(d) for d,_ in flows]; vals=np.array([v for _,v in flows]); tt=np.array([(d-ds[0]).days/365.2425 for d in ds])
    def f(x): return np.sum(vals/(1+x)**tt)
    lo,hi=-.999,10.; flo,fhi=f(lo),f(hi)
    for _ in range(200):
        if np.sign(flo)!=np.sign(fhi): break
        hi*=2; fhi=f(hi)
    for _ in range(200):
        m=(lo+hi)/2; fm=f(m)
        if np.sign(fm)==np.sign(flo): lo=m; flo=fm
        else: hi=m
    return (lo+hi)/2

def perf(o,diag=None):
    r=o['return']; nav=(1+r).cumprod(); yrs=(r.index[-1]-r.index[0]).days/365.2425
    cagr=float(nav.iloc[-1]**(1/yrs)-1); dd=nav/nav.cummax()-1; mdd=float(dd.min()); sd=float(r.std(ddof=1))
    out={'cagr':cagr,'max_drawdown':mdd,'sharpe':float(r.mean()/sd*np.sqrt(252)) if sd>0 else np.nan,'calmar':cagr/abs(mdd) if mdd<0 else np.nan,'final':float(o.value.iloc[-1]),'xirr':float(xirr(o)),'old_final':float(o.old_value.iloc[-1]),'flow_final':float(o.flow_value.iloc[-1]),'total_cost':float(o.cost.sum())}
    if diag: out.update(diag)
    return out

def main():
    od=Path('frequency_results'); od.mkdir(exist_ok=True)
    px=load_prices(Path('data')).loc[:END]; targets,meta=strategy(px)
    q=run_qqq(px); cases={'QQQ_all':(q,None,None,{'decision_count':0,'rebalance_count':0,'annual_turnover_notional_to_avg_nav':0.,'total_cost':float(q.cost.sum())})}
    for f in ['monthly','biweekly','weekly']:
        cases[f]=simulate(px,targets,meta,f)
    rows=[]; crisis={}; rolling={}
    for name,(o,tr,fl,dg) in cases.items():
        rows.append({'variant':name,**perf(o,dg)}); crisis[name]=crisis_stats(o['return']); o.to_csv(od/f'account_{name}.csv')
        if tr is not None: tr.to_csv(od/f'trades_{name}.csv',index=False)
        if fl is not None: fl.to_csv(od/f'flow_{name}.csv')
    summary=pd.DataFrame(rows); summary.to_csv(od/'summary.csv',index=False)
    for name,(o,_,__,___) in cases.items():
        if name=='QQQ_all': continue
        rolling[name]={'3y_vs_QQQ':rolling_summary(o['return'],q['return'],3),'5y_vs_QQQ':rolling_summary(o['return'],q['return'],5)}
    (od/'crisis_stats.json').write_text(json.dumps(crisis,indent=2)); (od/'rolling.json').write_text(json.dumps(rolling,indent=2))
    (od/'spec.json').write_text(json.dumps({'period':[START,END],'initial':INIT,'monthly_contribution':MONTHLY,'transaction_cost':TC,'soft_rebalance_band':BAND,'max_exposure':1.34,'TQQQ':False,'old_money':'Frozen V9 Regime/Kelly/RS; frequency is the only tested variable','new_money':'Original V9 dynamic flow: Regime base 50/60/80/95/100 + drawdown bonus + recovery/RS QQQ-SPY tilt; identical across frequencies','execution':'decision at period close -> next trading day; daily real/synthetic LETF paths'},indent=2))
    print(summary.to_string(index=False)); print(json.dumps(crisis,indent=2)); print(json.dumps(rolling,indent=2))

if __name__=='__main__': main()

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


def eff(w):
    return float(w.QQQ+w.SPY+2*w.QLD+2*w.SSO+3*w.TQQQ)

def norm(v):
    s=float(v.sum())
    return v/s if s>0 else pd.Series(0.,index=ASSETS)

def mark(v,p,prev):
    if prev is None: return v
    rr=(p/prev).replace([np.inf,-np.inf],np.nan).fillna(1.0)
    return v*rr

def need_rebalance(v,target):
    w=norm(v); ce,te=eff(w),eff(target)
    lev=float(w.QLD+w.SSO+w.TQQQ)
    if te<=1.0 and lev>.01: return True
    return abs(ce-te)>=BAND

def rebalance(v,target):
    target=target.reindex(ASSETS).fillna(0.)
    target=target/target.sum()
    total=float(v.sum()); desired=total*target
    notional=float((desired-v).abs().sum())
    cost=TC*notional
    return (total-cost)*target,notional,cost

def dd_bonus(dd):
    if not np.isfinite(dd): return 0.0
    if dd<=-.40: return .40
    if dd<=-.30: return .30
    if dd<=-.20: return .20
    if dd<=-.10: return .10
    return 0.0

def flow_target(meta_row, use_bonus=True):
    regime=str(meta_row['regime']) if pd.notna(meta_row['regime']) else 'Neutral'
    dd=float(meta_row['drawdown']) if pd.notna(meta_row['drawdown']) else 0.0
    rs=float(meta_row['rs126']) if pd.notna(meta_row['rs126']) else 0.0
    mom63=float(meta_row['mom63']) if pd.notna(meta_row['mom63']) else 0.0
    q=float(meta_row['q']) if 'q' in meta_row and pd.notna(meta_row['q']) else np.nan
    ma105=float(meta_row['ma105']) if pd.notna(meta_row['ma105']) else np.nan
    recovery=(dd<=-.20) and (mom63>0) and np.isfinite(q) and np.isfinite(ma105) and (q>ma105)
    eq=min(1.0, BASE_EQ.get(regime,.80) + (dd_bonus(dd) if use_bonus else 0.0))
    if recovery and dd<=-.30: qshare=.70
    elif recovery: qshare=.65
    elif regime=='StrongRiskOn': qshare=.70 if rs>0 else .55
    elif regime=='RiskOn': qshare=.65 if rs>0 else .50
    elif regime=='Neutral': qshare=.60 if rs>0 else .50
    else: qshare=.40
    t=pd.Series(0.,index=ASSETS)
    t.QQQ=eq*qshare; t.SPY=eq*(1-qshare); t.CASH=1-eq
    return t,eq,qshare,recovery,dd_bonus(dd) if use_bonus else 0.0

def buy_flow(flow,amount,target):
    a=amount*target
    cost=TC*float(a.QQQ+a.SPY)
    if a.CASH>=cost: a.CASH-=cost
    else:
        scale=max(0.,(a.sum()-cost)/a.sum())
        a*=scale
    return flow+a,cost

def make_strategy(px,variant):
    c=V9FinalConfig()
    if variant=='134_no_tqqq':
        c=replace(c,tqqq_capital_cap=0.,strong_trend_boost=0.,final_exposure_cap=1.34,strong_band=(1.15,1.40))
    elif variant=='134_tqqq':
        c=replace(c,strong_trend_boost=0.,final_exposure_cap=1.34,tqqq_capital_cap=.10,strong_band=(1.15,1.40))
    elif variant=='145_tqqq':
        c=replace(c,final_exposure_cap=1.45,tqqq_capital_cap=.10,strong_trend_boost=.11,strong_band=(1.15,1.40))
    else: raise ValueError(variant)
    st=V9FinalStrategy(c)
    sig,audit=st.generate_signals(px)
    _,meta=st.daily_targets(px)
    meta=meta.copy(); meta['q']=px['QQQ']
    return sig.reindex(columns=ASSETS),meta

def run_dynamic(px,variant,use_bonus=True):
    sig,meta=make_strategy(px,variant)
    dates=px.loc[START:END].index
    known=meta.shift(1).reindex(dates).ffill()  # t-1 information for first-trading-day contribution
    old=pd.Series(0.,index=ASSETS); flow=pd.Series(0.,index=ASSETS)
    prevp=None; prev_total=0.; last_month=None; initialized=False
    rows=[]; trades=[]; flows=[]
    for d in dates:
        p=px.loc[d,ASSETS]
        old=mark(old,p,prevp); flow=mark(flow,p,prevp)
        if not initialized:
            hist=sig.loc[:d]
            target=hist.iloc[-1] if len(hist) else sig.iloc[0]
            old=INIT*target/target.sum(); init_cost=TC*INIT; old*=((INIT-init_cost)/old.sum()); initialized=True
        contrib=0.; daycost=0.; eq=np.nan; qs=np.nan; rec=False; bonus=np.nan
        per=d.to_period('M')
        if per!=last_month:
            contrib=MONTHLY; last_month=per
            ft,eq,qs,rec,bonus=flow_target(known.loc[d],use_bonus)
            flow,c=buy_flow(flow,contrib,ft); daycost+=c
            flows.append((d,eq,qs,rec,bonus,float(known.loc[d,'drawdown']),str(known.loc[d,'regime'])))
        if d in sig.index and need_rebalance(old,sig.loc[d]):
            old,n,c=rebalance(old,sig.loc[d]); daycost+=c; trades.append((d,n,c,eff(norm(old)),eff(sig.loc[d])))
        total=float(old.sum()+flow.sum())
        r=(total-contrib)/prev_total-1 if prev_total>0 else 0.
        rows.append((r,total,float(old.sum()),float(flow.sum()),contrib,daycost,eff(norm(old))))
        prev_total=total; prevp=p
    out=pd.DataFrame(rows,index=dates,columns=['return','value','old_value','flow_value','contribution','cost','old_exposure'])
    tr=pd.DataFrame(trades,columns=['date','notional','cost','post_exposure','target_exposure'])
    fl=pd.DataFrame(flows,columns=['date','new_equity_ratio','new_qqq_share','recovery','dd_bonus','drawdown','regime']).set_index('date')
    return out,tr,fl

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
    ds=[pd.Timestamp(d) for d,_ in flows]; vals=np.array([v for _,v in flows]); tt=np.array([(d-ds[0]).days/365.2425 for d in ds])
    def f(x): return np.sum(vals/(1+x)**tt)
    lo,hi=-.999,10.; fl,fh=f(lo),f(hi)
    for _ in range(200):
        if np.sign(fl)!=np.sign(fh): break
        hi*=2; fh=f(hi)
    for _ in range(200):
        m=(lo+hi)/2; fm=f(m)
        if np.sign(fm)==np.sign(fl): lo=m; fl=fm
        else: hi=m
    return (lo+hi)/2

def perf(o,tr=None):
    r=o['return']; nav=(1+r).cumprod(); yrs=(r.index[-1]-r.index[0]).days/365.2425
    cagr=float(nav.iloc[-1]**(1/yrs)-1); dd=nav/nav.cummax()-1; mdd=float(dd.min()); sd=float(r.std(ddof=1))
    turn=float(tr.notional.sum()/o.value.mean()/yrs) if tr is not None and len(tr) else 0.
    return {'cagr':cagr,'max_drawdown':mdd,'sharpe':float(r.mean()/sd*np.sqrt(252)) if sd>0 else np.nan,'calmar':cagr/abs(mdd) if mdd<0 else np.nan,'final':float(o.value.iloc[-1]),'net_profit':float(o.value.iloc[-1]-(INIT+MONTHLY*len(o.index.to_period('M').unique()))),'xirr':float(xirr(o)),'old_final':float(o.old_value.iloc[-1]),'flow_final':float(o.flow_value.iloc[-1]),'annual_rebalance_notional_to_avg_nav':turn,'total_cost':float(o.cost.sum()),'max_old_exposure':float(o.old_exposure.max())}

def main():
    od=Path('v9_original_dynamic_results'); od.mkdir(exist_ok=True)
    px=load_prices(Path('data')).loc[:END]
    q=run_qqq(px)
    cases={'QQQ_all':(q,None,None)}
    for variant in ['134_no_tqqq','134_tqqq','145_tqqq']:
        o,tr,fl=run_dynamic(px,variant,True); cases[f'V9_original_dynamic_{variant}']=(o,tr,fl)
    o,tr,fl=run_dynamic(px,'134_no_tqqq',False); cases['V9_dynamic_134_no_dd_bonus']=(o,tr,fl)
    rows=[]
    for name,(o,tr,fl) in cases.items():
        rows.append({'variant':name,**perf(o,tr)}); o.to_csv(od/f'account_{name}.csv')
        if tr is not None: tr.to_csv(od/f'trades_{name}.csv',index=False)
        if fl is not None: fl.to_csv(od/f'flow_alloc_{name}.csv')
    summary=pd.DataFrame(rows); summary.to_csv(od/'summary.csv',index=False)
    crisis={k:crisis_stats(o['return']) for k,(o,_,__) in cases.items()}; (od/'crisis_stats.json').write_text(json.dumps(crisis,indent=2))
    rolling={k:{'3y_vs_QQQ':rolling_summary(o['return'],q['return'],3),'5y_vs_QQQ':rolling_summary(o['return'],q['return'],5)} for k,(o,_,__) in cases.items() if k!='QQQ_all'}
    (od/'rolling.json').write_text(json.dumps(rolling,indent=2))
    (od/'spec.json').write_text(json.dumps({'period':[START,END],'initial':INIT,'monthly':MONTHLY,'transaction_cost':TC,'soft_rebalance_band':BAND,'new_money_base_equity':BASE_EQ,'drawdown_bonus':{'dd_gt_-10':0,'-10_to_-20':.10,'-20_to_-30':.20,'-30_to_-40':.30,'lte_-40':.40},'new_money_qqq_share':{'recovery_dd_lte_-30':.70,'recovery':.65,'StrongRiskOn_rs_pos':.70,'StrongRiskOn_rs_nonpos':.55,'RiskOn_rs_pos':.65,'RiskOn_rs_nonpos':.50,'Neutral_rs_pos':.60,'Neutral_rs_nonpos':.50,'RiskOff_or_Crisis':.40},'note':'Strict reconstruction of previously defined V9 dynamic new-money engine; no newly invented HY/NFCI macro thresholds.'},indent=2))
    print(summary.to_string(index=False)); print(json.dumps(crisis,indent=2))

if __name__=='__main__': main()

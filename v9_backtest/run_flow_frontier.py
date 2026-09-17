#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.strategies.v9_final_strategy import V9FinalStrategy, V9FinalConfig
from run_formal_backtest import load_prices, crisis_stats
from run_split_sleeve_backtest import ASSETS, START, END, INIT, MONTHLY, TC, BAND, eff, weights, rebalance, need_rebalance, mark, xirr

QQQ_WEIGHTS=[0.60,0.70,0.80,0.90,1.00]

def run_fixed_flow(px,sig,qqq_weight):
    dates=px.loc[START:END].index
    sig=sig.reindex(columns=ASSETS)
    old=pd.Series(0.,index=ASSETS)
    flow=pd.Series(0.,index=ASSETS)
    prevp=None; prev_total=0.; last=None; initialized=False
    rows=[]; trades=[]
    for d in dates:
        p=px.loc[d,ASSETS]
        old=mark(old,p,prevp); flow=mark(flow,p,prevp)
        if not initialized:
            target=sig.loc[:d].iloc[-1] if len(sig.loc[:d]) else sig.iloc[0]
            old=INIT*target/target.sum(); cost=TC*INIT; old*= (INIT-cost)/old.sum(); initialized=True
        contrib=0.; daycost=0.; per=d.to_period('M')
        if per!=last:
            contrib=MONTHLY; last=per
            a=pd.Series(0.,index=ASSETS)
            a.QQQ=contrib*qqq_weight; a.SPY=contrib*(1.0-qqq_weight)
            cost=TC*contrib
            a*=max(0.,(contrib-cost)/a.sum())
            flow+=a; daycost+=cost
        if d in sig.index and need_rebalance(old,sig.loc[d]):
            old,n,c=rebalance(old,sig.loc[d]); daycost+=c; trades.append((d,n,c,eff(weights(old)),eff(sig.loc[d])))
        total=float(old.sum()+flow.sum())
        r=(total-contrib)/prev_total-1 if prev_total>0 else 0.
        rows.append((r,total,float(old.sum()),float(flow.sum()),contrib,daycost,eff(weights(old))))
        prev_total=total; prevp=p
    out=pd.DataFrame(rows,index=dates,columns=['return','value','old_value','flow_value','contribution','cost','old_exposure'])
    tr=pd.DataFrame(trades,columns=['date','notional','cost','post_exposure','target_exposure'])
    return out,tr

def perf(o,tr):
    r=o['return']; nav=(1+r).cumprod(); yrs=(r.index[-1]-r.index[0]).days/365.2425
    c=float(nav.iloc[-1]**(1/yrs)-1); dd=nav/nav.cummax()-1; m=float(dd.min()); sd=r.std(ddof=1)
    turn=float(tr.notional.sum()/o.value.mean()/yrs) if len(tr) else 0.
    total_contrib=INIT+float(o.contribution.sum())
    return {
      'cagr':c,'max_drawdown':m,'sharpe':float(r.mean()/sd*np.sqrt(252)) if sd>0 else np.nan,
      'calmar':c/abs(m) if m<0 else np.nan,'final':float(o.value.iloc[-1]),'xirr':float(xirr(o)),
      'total_contributed':total_contrib,'wealth_multiple_on_contributed':float(o.value.iloc[-1]/total_contrib),
      'profit':float(o.value.iloc[-1]-total_contrib),'old_final':float(o.old_value.iloc[-1]),'flow_final':float(o.flow_value.iloc[-1]),
      'annual_rebalance_notional_to_avg_nav':turn,'total_cost':float(o.cost.sum())
    }

def main():
    od=Path('frontier_results'); od.mkdir(exist_ok=True)
    px=load_prices(Path('data')).loc[:END]
    cfg=replace(V9FinalConfig(),tqqq_capital_cap=0.,strong_trend_boost=0.,final_exposure_cap=1.34)
    st=V9FinalStrategy(cfg); sig,_=st.generate_signals(px)
    rows=[]; crises={}
    for qw in QQQ_WEIGHTS:
        o,tr=run_fixed_flow(px,sig,qw)
        name=f'V9_old_plus_flow_QQQ_{int(qw*100)}'
        rows.append({'variant':name,'flow_qqq_weight':qw,**perf(o,tr)})
        crises[name]=crisis_stats(o['return'])
        o.to_csv(od/f'account_{int(qw*100)}.csv'); tr.to_csv(od/f'trades_{int(qw*100)}.csv',index=False)
    df=pd.DataFrame(rows)
    df.to_csv(od/'summary.csv',index=False)
    (od/'crisis_stats.json').write_text(json.dumps(crises,indent=2))
    (od/'diagnostics.json').write_text(json.dumps({'initial':INIT,'monthly':MONTHLY,'total_contributed':float(INIT+MONTHLY*312),'transaction_cost':TC,'soft_band':BAND,'flow_qqq_weights':QQQ_WEIGHTS,'old_sleeve':'V9 1.34x soft-rebalance frozen','flow_sleeve':'no-sell monthly QQQ/SPY fixed mix'},indent=2))
    print(df.to_string(index=False)); print(json.dumps(crises,indent=2))

if __name__=='__main__': main()

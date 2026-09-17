#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from run_formal_backtest import load_prices, crisis_stats
from run_frequency_backtest import ASSETS,START,END,INIT,MONTHLY,TC,strategy,flow_target,buy_flow,mark,norm,eff,rebalance,need_rebalance,make_schedule,perf,run_qqq

RULES={
 'base':(),
 'below175':('below175',),
 'vol35':('vol35',),
 'dd12':('dd12',),
 'vol_or_dd':('vol35','dd12'),
 'any_existing':('below175','vol35','dd12'),
}

def fire(rule,m):
    tests={
      'below175': pd.notna(m['ma175']) and pd.notna(m['q']) and float(m['q'])<float(m['ma175']),
      'vol35': pd.notna(m['vol20']) and float(m['vol20'])>0.35,
      'dd12': pd.notna(m['drawdown']) and float(m['drawdown'])<-0.12,
    }
    return any(tests[x] for x in rule),[k for k,v in tests.items() if v and k in rule]

def clamp(m):
    qs=float(m['qqq_share']) if pd.notna(m['qqq_share']) else .5; e=min(1.,float(m['exposure']) if pd.notna(m['exposure']) else 1.)
    t=pd.Series(0.,index=ASSETS); t.QQQ=e*qs; t.SPY=e*(1-qs); t.CASH=1-e; return t

def sim(px,targets,meta,rule):
    sched=make_schedule(px,targets,'monthly'); dates=px.loc[START:END].index; known=meta.shift(1).reindex(dates).ffill()
    old=pd.Series(0.,index=ASSETS); flow=pd.Series(0.,index=ASSETS); prevp=None; prev_total=0.; last=None; initialized=False; active=(sched.loc[:dates[0]].iloc[-1] if len(sched.loc[:dates[0]]) else sched.iloc[0]).copy(); latch=False; curm=None
    rows=[]; trades=[]; ec=0; total_cost=0.
    for d in dates:
        p=px.loc[d,ASSETS]; old=mark(old,p,prevp); flow=mark(flow,p,prevp); per=d.to_period('M')
        if curm is None or per!=curm: curm=per; latch=False
        if not initialized:
            old=INIT*active/active.sum(); c=TC*INIT; old*=((INIT-c)/old.sum()); total_cost+=c; initialized=True
        contrib=0.; daycost=0.
        if per!=last:
            contrib=MONTHLY; last=per; ft,*_=flow_target(known.loc[d]); flow,c=buy_flow(flow,contrib,ft); total_cost+=c; daycost+=c
        if d in sched.index:
            active=sched.loc[d].copy(); latch=False
            if need_rebalance(old,active):
                old,n,c=rebalance(old,active); trades.append((d,'monthly',n)); total_cost+=c; daycost+=c
        if rule and (not latch) and old.sum()>0 and float(norm(old)[['QLD','SSO','TQQQ']].sum())>.01:
            ok,reasons=fire(rule,known.loc[d])
            if ok:
                old,n,c=rebalance(old,clamp(known.loc[d])); trades.append((d,'emergency',n)); total_cost+=c; daycost+=c; latch=True; ec+=1
        total=float(old.sum()+flow.sum()); r=(total-contrib)/prev_total-1 if prev_total>0 else 0.; rows.append((d,r,total,float(old.sum()),float(flow.sum()),contrib,daycost,eff(norm(old)))); prev_total=total; prevp=p
    o=pd.DataFrame(rows,columns=['date','return','value','old_value','flow_value','contribution','cost','old_exposure']).set_index('date'); tr=pd.DataFrame(trades,columns=['date','kind','notional']); yrs=(o.index[-1]-o.index[0]).days/365.2425
    dg={'decision_count':int(len(sched.loc[START:END])),'rebalance_count':int((tr.kind=='monthly').sum()) if len(tr) else 0,'emergency_count':ec,'annual_turnover_notional_to_avg_nav':float(tr.notional.sum()/o.value.mean()/yrs) if len(tr) else 0.,'total_cost':float(total_cost)}
    return o,dg

def main():
    od=Path('emergency_ablation_results'); od.mkdir(exist_ok=True); px=load_prices(Path('data')).loc[:END]; targets,meta=strategy(px); meta=meta.copy(); meta['q']=px.QQQ
    rows=[]; crisis={}
    for name,rule in RULES.items():
        o,dg=sim(px,targets,meta,rule); rows.append({'variant':name,**perf(o,dg)}); crisis[name]=crisis_stats(o['return']); o.to_csv(od/f'account_{name}.csv')
    s=pd.DataFrame(rows); s.to_csv(od/'summary.csv',index=False); (od/'crisis_stats.json').write_text(json.dumps(crisis,indent=2)); (od/'spec.json').write_text(json.dumps({'rules':{k:list(v) for k,v in RULES.items()},'thresholds':{'below175':'QQQ<175DMA','vol35':'20D realized vol >35%','dd12':'QQQ drawdown<-12%'},'note':'Ablation only; thresholds unchanged, no tuning.'},indent=2)); print(s.to_string(index=False)); print(json.dumps(crisis,indent=2))
if __name__=='__main__': main()

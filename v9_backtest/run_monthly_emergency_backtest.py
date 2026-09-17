#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from run_formal_backtest import load_prices, crisis_stats, rolling_summary
from run_frequency_backtest import (
    ASSETS, START, END, INIT, MONTHLY, TC, BAND,
    strategy, flow_target, buy_flow, mark, norm, eff, rebalance, need_rebalance,
    make_schedule, xirr, perf, run_qqq
)


def clamp_to_one(active: pd.Series, meta_row: pd.Series) -> pd.Series:
    """Emergency deleverage only: remove 2x/3x ETFs and cap effective exposure at 1.0x."""
    qs=float(meta_row['qqq_share']) if pd.notna(meta_row['qqq_share']) else 0.5
    e=min(1.0, float(meta_row['exposure']) if pd.notna(meta_row['exposure']) else 1.0)
    t=pd.Series(0.0,index=ASSETS)
    t['QQQ']=e*qs; t['SPY']=e*(1-qs); t['CASH']=1-e
    return t


def simulate_monthly(px,targets,meta,emergency=False):
    schedule=make_schedule(px,targets,'monthly')
    dates=px.loc[START:END].index
    known=meta.shift(1).reindex(dates).ffill()
    old=pd.Series(0.,index=ASSETS); flow=pd.Series(0.,index=ASSETS)
    prevp=None; prev_total=0.; last_month=None; initialized=False
    rows=[]; trades=[]; flowlog=[]; emergency_log=[]; total_cost=0.
    active_pre=schedule.loc[:dates[0]]
    active=(active_pre.iloc[-1] if len(active_pre) else schedule.iloc[0]).copy()
    emergency_latched=False
    current_month=None

    for d in dates:
        p=px.loc[d,ASSETS]; old=mark(old,p,prevp); flow=mark(flow,p,prevp)
        per=d.to_period('M')
        if current_month is None or per!=current_month:
            current_month=per; emergency_latched=False
        if not initialized:
            old=INIT*active/active.sum(); init_cost=TC*INIT; old*=((INIT-init_cost)/old.sum()); total_cost+=init_cost; initialized=True

        contrib=0.; daycost=0.
        if per!=last_month:
            contrib=MONTHLY; last_month=per
            ft,eq,qs,rec=flow_target(known.loc[d]); flow,c=buy_flow(flow,contrib,ft); daycost+=c; total_cost+=c
            flowlog.append((d,eq,qs,rec,float(known.loc[d,'drawdown']),str(known.loc[d,'regime'])))

        if d in schedule.index:
            active=schedule.loc[d].copy(); emergency_latched=False
            if need_rebalance(old,active):
                old,n,c=rebalance(old,active); trades.append((d,'monthly',n,c,eff(norm(old)),eff(active))); daycost+=c; total_cost+=c

        if emergency and (not emergency_latched) and old.sum()>0:
            w=norm(old); lev=float(w[['QLD','SSO','TQQQ']].sum())
            m=known.loc[d]
            trigger=(lev>.01) and (
                (pd.notna(m['ma175']) and pd.notna(m['q']) and float(m['q']) < float(m['ma175'])) or
                (pd.notna(m['vol20']) and float(m['vol20']) > 0.35) or
                (pd.notna(m['drawdown']) and float(m['drawdown']) < -0.12)
            )
            if trigger:
                target=clamp_to_one(active,m)
                old,n,c=rebalance(old,target); daycost+=c; total_cost+=c; emergency_latched=True
                trades.append((d,'emergency',n,c,eff(norm(old)),eff(target)))
                reasons=[]
                if pd.notna(m['ma175']) and pd.notna(m['q']) and float(m['q']) < float(m['ma175']): reasons.append('below175')
                if pd.notna(m['vol20']) and float(m['vol20']) > 0.35: reasons.append('vol20>35')
                if pd.notna(m['drawdown']) and float(m['drawdown']) < -0.12: reasons.append('dd<-12')
                emergency_log.append((d,'+'.join(reasons),float(m['drawdown']),float(m['vol20']) if pd.notna(m['vol20']) else np.nan,float(m['q']) if pd.notna(m['q']) else np.nan,float(m['ma175']) if pd.notna(m['ma175']) else np.nan))

        total=float(old.sum()+flow.sum())
        r=(total-contrib)/prev_total-1 if prev_total>0 else 0.
        rows.append((d,r,total,float(old.sum()),float(flow.sum()),contrib,daycost,eff(norm(old))))
        prev_total=total; prevp=p

    out=pd.DataFrame(rows,columns=['date','return','value','old_value','flow_value','contribution','cost','old_exposure']).set_index('date')
    tr=pd.DataFrame(trades,columns=['date','kind','notional','cost','post_exposure','target_exposure'])
    fl=pd.DataFrame(flowlog,columns=['date','new_equity_ratio','new_qqq_share','recovery','drawdown','regime']).set_index('date')
    em=pd.DataFrame(emergency_log,columns=['date','reason','drawdown','vol20','q','ma175'])
    yrs=(out.index[-1]-out.index[0]).days/365.2425
    turnover=float(tr.notional.sum()/out.value.mean()/yrs) if len(tr) else 0.
    diag={'decision_count':int(len(schedule.loc[START:END])),'rebalance_count':int((tr.kind=='monthly').sum()) if len(tr) else 0,'emergency_count':int((tr.kind=='emergency').sum()) if len(tr) else 0,'annual_turnover_notional_to_avg_nav':turnover,'total_cost':float(total_cost)}
    return out,tr,fl,em,diag


def main():
    od=Path('emergency_results'); od.mkdir(exist_ok=True)
    px=load_prices(Path('data')).loc[:END]; targets,meta=strategy(px); meta=meta.copy(); meta['q']=px['QQQ']
    q=run_qqq(px)
    cases={'QQQ_all':(q,None,None,None,{'decision_count':0,'rebalance_count':0,'emergency_count':0,'annual_turnover_notional_to_avg_nav':0.,'total_cost':float(q.cost.sum())})}
    cases['monthly_base']=simulate_monthly(px,targets,meta,False)
    cases['monthly_emergency']=simulate_monthly(px,targets,meta,True)
    rows=[]; crisis={}; rolling={}
    for name,(o,tr,fl,em,dg) in cases.items():
        rows.append({'variant':name,**perf(o,dg)}); crisis[name]=crisis_stats(o['return']); o.to_csv(od/f'account_{name}.csv')
        if tr is not None: tr.to_csv(od/f'trades_{name}.csv',index=False)
        if fl is not None: fl.to_csv(od/f'flow_{name}.csv')
        if em is not None: em.to_csv(od/f'emergency_{name}.csv',index=False)
    summary=pd.DataFrame(rows); summary.to_csv(od/'summary.csv',index=False)
    for name,(o,_,__,___,____) in cases.items():
        if name=='QQQ_all': continue
        rolling[name]={'3y_vs_QQQ':rolling_summary(o['return'],q['return'],3),'5y_vs_QQQ':rolling_summary(o['return'],q['return'],5)}
    (od/'crisis_stats.json').write_text(json.dumps(crisis,indent=2)); (od/'rolling.json').write_text(json.dumps(rolling,indent=2))
    (od/'spec.json').write_text(json.dumps({'period':[START,END],'initial':INIT,'monthly_contribution':MONTHLY,'transaction_cost':TC,'soft_rebalance_band':BAND,'max_exposure':1.34,'TQQQ':False,'strategic_frequency':'monthly','emergency_rule':'only while old-money sleeve holds leverage: if QQQ<175DMA OR vol20>35% OR drawdown<-12%, immediately rebalance old-money sleeve to <=1.0x; no re-leverage until next monthly decision','new_money':'unchanged original V9 dynamic flow'},indent=2))
    print(summary.to_string(index=False)); print(json.dumps(crisis,indent=2)); print(json.dumps(rolling,indent=2))

if __name__=='__main__': main()

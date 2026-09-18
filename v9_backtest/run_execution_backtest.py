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
TC=0.001
MONTHLY=20_000.0
START='2000-01-03'; END='2025-12-31'
BAND=0.05
NEW_EQ={'Crisis':0.50,'RiskOff':0.60,'Neutral':0.80,'RiskOn':0.95,'StrongRiskOn':1.00}


def eff(w):
    return float(w.get('QQQ',0)+w.get('SPY',0)+2*w.get('QLD',0)+2*w.get('SSO',0)+3*w.get('TQQQ',0))

def current_weights(values):
    s=float(values.sum())
    return values/s if s>0 else pd.Series(0.0,index=ASSETS)

def full_rebalance(values,target,tc=TC):
    total=float(values.sum()); target=target.reindex(ASSETS).fillna(0.0)
    target=target/target.sum() if target.sum()>0 else pd.Series({'CASH':1.0}).reindex(ASSETS).fillna(0.0)
    desired=total*target
    notional=float((desired-values).abs().sum())
    cost=tc*notional
    total2=max(0.0,total-cost)
    return total2*target, notional, cost

def add_contribution(values, amount, mode, target, regime, qshare, tc=TC):
    # Contribution purchases incur realistic one-way transaction costs.
    if mode in ('exact','soft_unified'):
        post=float(values.sum())+amount
        desired=post*target.reindex(ASSETS).fillna(0.0)
        deficit=(desired-values).clip(lower=0.0)
        if deficit.sum()>0:
            alloc=amount*deficit/deficit.sum()
        else:
            alloc=pd.Series(0.0,index=ASSETS); alloc['CASH']=amount
    else:
        eqr=1.0 if mode=='soft_split100' else NEW_EQ.get(str(regime),0.80)
        eqamt=amount*eqr; alloc=pd.Series(0.0,index=ASSETS)
        # New money never buys leverage. Tilt QQQ/SPY toward the underweight side.
        cur_eq=float(values['QQQ']+values['SPY'])
        cur_qshare=float(values['QQQ']/cur_eq) if cur_eq>0 else qshare
        if cur_qshare < qshare-0.02:
            qalloc=eqamt
        elif cur_qshare > qshare+0.02:
            qalloc=0.0
        else:
            qalloc=eqamt*qshare
        alloc['QQQ']=qalloc; alloc['SPY']=eqamt-qalloc; alloc['CASH']=amount-eqamt
    traded=float(alloc.drop('CASH').sum())
    cost=tc*traded
    values=values+alloc
    # Pay contribution trade cost from cash when possible; otherwise pro-rata portfolio.
    if values['CASH']>=cost: values['CASH']-=cost
    elif values.sum()>0: values*=max(0.0,(values.sum()-cost)/values.sum())
    return values,traded,cost

def should_rebalance(values,target,mode):
    if mode=='exact': return True
    w=current_weights(values); ce=eff(w); te=eff(target)
    lev_cur=float(w['QLD']+w['SSO']+w['TQQQ']); lev_tgt=float(target['QLD']+target['SSO']+target['TQQQ'])
    # Always honor meaningful de-leveraging / TQQQ exits.
    if te<=1.0 and lev_cur>0.01: return True
    if w['TQQQ']>0.005 and target['TQQQ']<0.001: return True
    if ce-te >= BAND: return True
    if te-ce >= BAND: return True
    # Large composition drift also triggers; keeps QQQ/SPY/2x implementation sane.
    if float((w-target).abs().max()) >= 0.10: return True
    return False

def simulate(px, signals, meta, mode, monthly=MONTHLY, tc=TC):
    dates=px.loc[START:END].index
    sig=signals.reindex(columns=ASSETS).copy()
    # latest executable target known on each day
    targets=sig.reindex(dates).ffill()
    first_target=sig.iloc[0] if len(sig) else pd.Series({'CASH':1.0},index=ASSETS).fillna(0.0)
    targets=targets.fillna(first_target)
    lagmeta=meta.shift(1).reindex(dates).ffill()
    values=pd.Series(0.0,index=ASSETS)
    prev_prices=None; prev_total=0.0
    twr=[]; vals=[]; turnovers=[]; costs=[]; contribs=[]; exposure=[]; trades=[]
    last_period=None
    for d in dates:
        p=px.loc[d,ASSETS]
        # Mark holdings to market from previous close.
        if prev_prices is not None:
            rr=(p/prev_prices).replace([np.inf,-np.inf],np.nan).fillna(1.0)
            values=values*rr
        before_flow=float(values.sum())
        contribution=0.0; day_turn=0.0; day_cost=0.0
        per=d.to_period('M')
        if per!=last_period:
            contribution=monthly; last_period=per
            target=targets.loc[d]
            regime=lagmeta.loc[d,'regime'] if 'regime' in lagmeta else 'Neutral'
            qshare=float(lagmeta.loc[d,'qqq_share']) if 'qqq_share' in lagmeta and pd.notna(lagmeta.loc[d,'qqq_share']) else 0.5
            values,tr,c=add_contribution(values,contribution,mode,target,regime,qshare,tc)
            day_turn+=tr; day_cost+=c
        # Execute target only on actual strategy signal dates.
        if d in sig.index:
            target=sig.loc[d]
            if should_rebalance(values,target,mode):
                pre=float(values.sum()); values,tr,c=full_rebalance(values,target,tc)
                day_turn+=tr; day_cost+=c; trades.append((d,mode,'rebalance',tr,c,eff(current_weights(values)),eff(target)))
        end_total=float(values.sum())
        if prev_total>0:
            r=(end_total-contribution)/prev_total-1.0
        else:
            r=0.0
        twr.append(r); vals.append(end_total); turnovers.append(day_turn/max(end_total,1e-12)); costs.append(day_cost); contribs.append(contribution); exposure.append(eff(current_weights(values)))
        prev_total=end_total; prev_prices=p
    out=pd.DataFrame({'return':twr,'value':vals,'turnover':turnovers,'cost':costs,'contribution':contribs,'exposure':exposure},index=dates)
    return out,pd.DataFrame(trades,columns=['date','mode','action','notional','cost','post_exposure','target_exposure'])

def xirr_from_account(out):
    flows=[]
    for d,x in out.loc[out.contribution>0,'contribution'].items(): flows.append((d,-float(x)))
    flows.append((out.index[-1],float(out.value.iloc[-1])))
    dates=[pd.Timestamp(d) for d,_ in flows]; vals=np.array([v for _,v in flows],float); t=np.array([(d-dates[0]).days/365.2425 for d in dates])
    def f(rate): return np.sum(vals/np.power(1+rate,t))
    lo,hi=-.999,10.; flo,fhi=f(lo),f(hi)
    for _ in range(200):
        if np.sign(flo)!=np.sign(fhi): break
        hi*=2; fhi=f(hi)
    if np.sign(flo)==np.sign(fhi): return np.nan
    for _ in range(200):
        m=(lo+hi)/2; fm=f(m)
        if np.sign(fm)==np.sign(flo): lo=m; flo=fm
        else: hi=m
    return (lo+hi)/2

def perf(out):
    r=out['return']; nav=(1+r).cumprod(); years=(r.index[-1]-r.index[0]).days/365.2425
    cagr=float(nav.iloc[-1]**(1/years)-1); dd=nav/nav.cummax()-1; mdd=float(dd.min()); vol=float(r.std(ddof=1)*np.sqrt(252)); sh=float(r.mean()/r.std(ddof=1)*np.sqrt(252)) if r.std()>0 else np.nan
    return {'cagr':cagr,'max_drawdown':mdd,'sharpe':sh,'calmar':cagr/abs(mdd) if mdd<0 else np.nan,'dca_final':float(out.value.iloc[-1]),'dca_xirr':float(xirr_from_account(out)),'annual_turnover':float(out.turnover.mean()*252),'total_cost':float(out.cost.sum()),'max_exposure':float(out.exposure.max())}

def qqq_dca(px,monthly=MONTHLY,tc=TC):
    dates=px.loc[START:END].index; values=0.; prev=None; last=None; rows=[]
    for d in dates:
        p=float(px.loc[d,'QQQ'])
        if prev is not None: values*=p/prev
        before=values; contrib=0.; cost=0.
        per=d.to_period('M')
        if per!=last:
            contrib=monthly; last=per; cost=tc*monthly; values+=monthly-cost
        r=(values-contrib)/before-1 if before>0 else 0.
        rows.append((r,values,(monthly/max(values,1e-12) if contrib else 0),cost,contrib,1.0)); prev=p
    return pd.DataFrame(rows,index=dates,columns=['return','value','turnover','cost','contribution','exposure'])

def main():
    outdir=Path('execution_results'); outdir.mkdir(exist_ok=True)
    px=load_prices(Path('data')).loc[:END]
    cfg=replace(V9FinalConfig(),tqqq_capital_cap=0.0,strong_trend_boost=0.0,final_exposure_cap=1.34)
    strat=V9FinalStrategy(cfg); sig,audit=strat.generate_signals(px); _,meta=strat.daily_targets(px)
    variants={}
    variants['QQQ_DCA']=qqq_dca(px)
    for mode in ['exact','soft_unified','soft_splitbase','soft_split100']:
        variants[mode],tr=simulate(px,sig,meta,mode)
        tr.to_csv(outdir/f'trades_{mode}.csv',index=False)
        variants[mode].to_csv(outdir/f'account_{mode}.csv')
    rows=[]
    for name,o in variants.items(): rows.append({'variant':name,**perf(o)})
    summary=pd.DataFrame(rows); summary.to_csv(outdir/'summary.csv',index=False)
    crises={k:crisis_stats(v['return']) for k,v in variants.items()}; (outdir/'crisis_stats.json').write_text(json.dumps(crises,indent=2))
    rolling={}
    for name,o in variants.items():
        if name=='QQQ_DCA': continue
        rolling[name]={'3y_vs_QQQ':rolling_summary(o['return'],variants['QQQ_DCA']['return'],3),'5y_vs_QQQ':rolling_summary(o['return'],variants['QQQ_DCA']['return'],5)}
    (outdir/'rolling.json').write_text(json.dumps(rolling,indent=2))
    diagnostics={'band':BAND,'tc':TC,'monthly':MONTHLY,'start':START,'end':END,'new_money_equity':NEW_EQ,'signal_count':int(len(sig)),'rows':int(len(px.loc[START:END]))}
    (outdir/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    print(summary.to_string(index=False)); print(json.dumps(crises,indent=2))
if __name__=='__main__': main()

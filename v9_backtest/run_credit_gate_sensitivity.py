#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

import run_frequency_backtest as rf
from run_formal_backtest import load_prices
import run_v10_research_experiments as r1

OUT=Path('credit_gate_sensitivity'); OUT.mkdir(exist_ok=True)
END='2025-12-31'


def load_oas(path,index):
    x=pd.read_csv(path); dc=x.columns[0]; vc=x.columns[1]
    x[dc]=pd.to_datetime(x[dc],errors='coerce'); x[vc]=pd.to_numeric(x[vc],errors='coerce')
    return x.dropna(subset=[dc]).set_index(dc)[vc].sort_index().reindex(index).ffill()

def run_gate(px,base_targets,base_meta,oas,years,q):
    win=int(round(252*years)); minp=int(round(win*2/3))
    thr=oas.rolling(win,min_periods=minp).quantile(q)
    stress=(oas.shift(1)>thr.shift(1)).fillna(False)
    targets=base_targets.copy(); base_e=base_targets.apply(rf.eff,axis=1)
    e=base_e.where(~stress,np.minimum(base_e,1.0))
    arr=[]
    for d in targets.index:
        arr.append(r1.target_from_exposure(e.loc[d],base_meta.loc[d,'qqq_share'],bool(base_meta.loc[d,'rs126']>0)).values)
    targets=pd.DataFrame(np.asarray(arr),index=targets.index,columns=rf.ASSETS)
    meta=base_meta.copy(); meta['credit_stress']=stress
    o,tr,fl,dg=rf.simulate(px,targets,meta,'monthly')
    return o,dg,stress,base_e

def main():
    px=load_prices(Path('data')).loc[:END]
    bt,bm=rf.strategy(px); oas=load_oas(Path('data/hy_oas_full.csv'),px.index)
    base,_,_,bdg=rf.simulate(px,bt,bm,'monthly'); bp=rf.perf(base,bdg)
    rows=[]
    for years in [2,3,5]:
        for q in [.85,.90,.95]:
            o,dg,stress,base_e=run_gate(px,bt,bm,oas,years,q); p=rf.perf(o,dg)
            md=rf.decision_dates(px,'monthly')
            rows.append({'lookback_years':years,'quantile':q,'final':p['final'],'xirr':p['xirr'],'cagr':p['cagr'],'mdd':p['max_drawdown'],'sharpe':p['sharpe'],'calmar':p['calmar'],'final_delta':p['final']-bp['final'],'xirr_delta':p['xirr']-bp['xirr'],'mdd_delta':p['max_drawdown']-bp['max_drawdown'],'sharpe_delta':p['sharpe']-bp['sharpe'],'stress_months':int(stress.loc[md].sum()),'cap_months':int((stress.loc[md] & (base_e.loc[md]>1)).sum()),'rebalance_count':dg['rebalance_count']})
    df=pd.DataFrame(rows); df.to_csv(OUT/'sensitivity.csv',index=False)
    diag={'baseline':bp,'positive_final_count':int((df.final_delta>0).sum()),'nonworse_mdd_count':int((df.mdd_delta>=-1e-12).sum()),'positive_sharpe_count':int((df.sharpe_delta>0).sum()),'all_three_count':int(((df.final_delta>0)&(df.mdd_delta>=-1e-12)&(df.sharpe_delta>0)).sum()),'grid':'lookback 2/3/5 years x percentile 85/90/95; predeclared neighborhood check, not parameter selection'}
    (OUT/'diagnostics.json').write_text(json.dumps(diag,indent=2))
    print(df.to_string(index=False)); print(json.dumps(diag,indent=2))
if __name__=='__main__': main()

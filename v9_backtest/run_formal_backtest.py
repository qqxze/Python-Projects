#!/usr/bin/env python3
from __future__ import annotations
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from src.strategies.v9_final_strategy import V9FinalStrategy, V9FinalConfig
from src.backtest.finrl_semantic_engine import BacktestConfig, run_weight_backtest, metrics, dca_account, xirr

ASSETS=['QQQ','SPY','QLD','SSO','TQQQ','CASH']


def load_two_col(path: Path, name: str, value_col='Close') -> pd.Series:
    df=pd.read_csv(path,sep='\t')
    dcol='Date' if 'Date' in df.columns else df.columns[0]
    ccol=value_col if value_col in df.columns else df.columns[-1]
    idx=pd.to_datetime(df[dcol]).dt.tz_localize(None).dt.normalize()
    s=pd.Series(pd.to_numeric(df[ccol],errors='coerce').values,index=idx,name=name).dropna()
    return s[~s.index.duplicated(keep='last')].sort_index()


def build_cash(rate_path: Path, trading_index: pd.DatetimeIndex) -> pd.Series:
    rates=load_two_col(rate_path,'RATE',value_col='Rate').reindex(trading_index).ffill().bfill()
    daily=(1.0 + rates/100.0) ** (1.0/252.0) - 1.0
    cash=(1.0+daily).cumprod()
    cash.iloc[0]=1.0
    return cash.rename('CASH')


def load_prices(data_dir: Path) -> pd.DataFrame:
    fmap={'QQQ':'synthetic-qqq.tsv','SPY':'spy.tsv','QLD':'synthetic-qld.tsv','SSO':'synthetic-sso.tsv','TQQQ':'synthetic-tqqq.tsv'}
    px=pd.concat([load_two_col(data_dir/fn,k) for k,fn in fmap.items()],axis=1).sort_index().ffill()
    px=px.dropna(subset=list(fmap))
    px['CASH']=build_cash(data_dir/'short-rates.tsv',px.index)
    return px[ASSETS]


def buyhold_signal(px: pd.DataFrame, weights: dict[str,float]) -> pd.DataFrame:
    d=px.index[0]
    s=pd.DataFrame([weights],index=[d]).reindex(columns=ASSETS,fill_value=0.0)
    return s


def strategy_signals(px, variant):
    c=V9FinalConfig()
    if variant=='v9_nolev':
        c=replace(c,strong_band=(1.0,1.0),riskon_band=(0.95,1.0),tqqq_capital_cap=0.0,strong_trend_boost=0.0,final_exposure_cap=1.0)
    elif variant=='v9_134':
        c=replace(c,tqqq_capital_cap=0.0,strong_trend_boost=0.0,final_exposure_cap=1.34)
    elif variant=='v9_tqqq_same134':
        c=replace(c,strong_trend_boost=0.0,final_exposure_cap=1.34,tqqq_capital_cap=0.10)
    elif variant=='v9_final145':
        pass
    else: raise ValueError(variant)
    return V9FinalStrategy(c).generate_signals(px)


def underwater_days(r: pd.Series) -> int:
    nav=(1+r).cumprod(); dd=nav/nav.cummax()-1; cur=best=0
    for x in dd<0:
        cur=cur+1 if x else 0; best=max(best,cur)
    return best


def crisis_stats(r):
    windows={'dotcom':['2000-03-24','2002-10-09'],'gfc':['2007-10-09','2009-03-09'],'covid':['2020-02-19','2020-03-23'],'bear2022':['2022-01-03','2022-10-14']}
    out={}
    for k,(a,b) in windows.items():
        x=r.loc[a:b]
        if len(x):
            n=(1+x).cumprod(); out[k]={'return':float(n.iloc[-1]-1),'max_drawdown':float((n/n.cummax()-1).min())}
    return out


def rolling_summary(r, bench, years):
    starts=pd.date_range(max(r.index.min(),bench.index.min()),min(r.index.max(),bench.index.max())-pd.DateOffset(years=years),freq='MS')
    diffs=[]; ddwins=[]
    for st in starts:
        en=st+pd.DateOffset(years=years); a=r.loc[st:en]; b=bench.loc[st:en]
        if len(a)<200*years or len(b)<200*years: continue
        diffs.append(float((1+a).prod()/(1+b).prod()-1))
        na=(1+a).cumprod(); nb=(1+b).cumprod(); dda=float((na/na.cummax()-1).min()); ddb=float((nb/nb.cummax()-1).min())
        ddwins.append(dda>ddb)
    return {'n':len(diffs),'beat_wealth_pct':float(np.mean(np.array(diffs)>0)) if diffs else np.nan,'median_wealth_excess':float(np.median(diffs)) if diffs else np.nan,'lower_mdd_pct':float(np.mean(ddwins)) if ddwins else np.nan}


def run(data_dir:Path,start='2000-01-03',end='2025-12-31',out_dir=Path('results'),monthly=20000.0):
    out_dir.mkdir(parents=True,exist_ok=True)
    px=load_prices(data_dir).loc[:end]
    variants={'QQQ':buyhold_signal(px,{'QQQ':1}),'QQQ_SPY_50_50':buyhold_signal(px,{'QQQ':.5,'SPY':.5})}
    audits={}
    for v in ['v9_nolev','v9_134','v9_tqqq_same134','v9_final145']:
        variants[v],audits[v]=strategy_signals(px,v)
    rows=[]; allr={}; navs={}
    for tc in [0.0005,0.0010,0.0020]:
        for name,sig in variants.items():
            bt=run_weight_backtest(px,sig,BacktestConfig(start,end,transaction_cost=tc)); m=metrics(bt['returns'],bt['turnover']); dca,flows=dca_account(bt['returns'],monthly); irr=xirr(flows)
            row={'variant':name,'transaction_cost':tc,**m,'dca_final':float(dca.iloc[-1]),'dca_xirr':float(irr),'underwater_days':underwater_days(bt['returns'])}
            rows.append(row)
            if tc==.001:
                allr[name]=bt['returns']; navs[name]=bt['nav']; bt['weights'].to_csv(out_dir/f'weights_{name}.csv'); bt['turnover'].rename('turnover').to_csv(out_dir/f'turnover_{name}.csv')
    summary=pd.DataFrame(rows); summary.to_csv(out_dir/'summary.csv',index=False); pd.DataFrame(navs).to_csv(out_dir/'navs_10bp.csv')
    (out_dir/'crisis_stats.json').write_text(json.dumps({k:crisis_stats(v) for k,v in allr.items()},indent=2),encoding='utf-8')
    rolling={}
    for name,r in allr.items():
        if name=='QQQ': continue
        rolling[name]={'3y_vs_QQQ':rolling_summary(r,allr['QQQ'],3),'5y_vs_QQQ':rolling_summary(r,allr['QQQ'],5)}
    (out_dir/'rolling.json').write_text(json.dumps(rolling,indent=2),encoding='utf-8')
    for name,a in audits.items(): a.to_csv(out_dir/f'audit_{name}.csv')
    # machine-readable diagnostics
    diag={
      'start':start,'end':end,'rows':int(len(px.loc[start:end])),
      'first_date':str(px.loc[start:end].index.min().date()),'last_date':str(px.loc[start:end].index.max().date()),
      'tqqq_max_weight':float(pd.read_csv(out_dir/'weights_v9_final145.csv',index_col=0)['TQQQ'].max()),
      'summary_rows':int(len(summary))
    }
    (out_dir/'diagnostics.json').write_text(json.dumps(diag,indent=2),encoding='utf-8')
    print(summary.to_string(index=False))

if __name__=='__main__':
    run(Path('data'))

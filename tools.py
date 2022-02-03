

import pandas as pd
import numpy as np
from woe5 import woe 

def ks(test,pred):
    data =  pd.DataFrame({'bad':test, 'score':pred})
    #data.replace({'bad': {0: 1,1:0}})
    data['good'] = 1 - data.bad
    data['bucket'] = pd.qcut(data.score, 100,duplicates='drop')
    grouped = data.groupby('bucket', as_index = False) 
    agg1 = grouped.min().score
    agg1 = pd.DataFrame(grouped.min().score)
    agg1=agg1.rename(columns={'score': 'min_scr'})
    agg1['max_scr'] = grouped.max().score
    agg1['bads'] = grouped.sum().bad
    agg1['goods'] = grouped.sum().good 
    agg1['total'] = agg1.bads + agg1.goods
    agg2 = (agg1.sort_index(by = 'min_scr')).reset_index(drop = True)
    agg2['odds'] = (agg2.goods / agg2.bads).apply('{0:.2f}'.format)
    agg2['bad_rate'] = (agg2.bads / agg2.total).apply('{0:.2%}'.format)
 # CALCULATE KS STATISTIC
    agg2['ks'] = np.round(((agg2.bads / data.bad.sum()).cumsum() - (agg2.goods / data.good.sum()).cumsum()), 4) * 100
 # DEFINE A FUNCTION TO FLAG MAX KS 
    flag = lambda x: '<----' if x == agg2.ks.min() else '' 
# FLAG OUT MAX KS 
    agg2['max_ks'] = agg2.ks.apply(flag)
    return abs(agg2['ks'].min()),agg2


def deploy_frame(frame,df):
    pre_bins=list(frame['breaks'])
    if pre_bins[-1]==0:
        pre_bins.pop()
    bins=[-float('Inf')]
    bins.extend(pre_bins)
    name=frame.iloc[0,11]
    w=woe(bins=bins,name=name,stat=frame)
    df[name+'_binned']=w.deploy(df)

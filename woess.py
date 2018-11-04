"""
Created on Sat Nov  3 10:13:22 2018

@author: pc
"""

import pandas as pd
import numpy as np

class met:
    def __init__(self,bins=None,breaks=5):
        self.bins=bins if bins is None else bins
        self.stat=None 
        self.iv=None
        self.breaks=breaks
        self.name=None

    def fit(self,x,y):
        '''Fitting Information'''
        x = pd.Series(x)
        y = pd.Series(y)
        self.name=x.name
        df = pd.DataFrame({"X": x, "Y": y, 'order': np.arange(x.size)})
        
#        bins=[]
#        bins.append(-float('Inf'))
#        
#        breaks = self.breaks
#        br=1/breaks
#        quant=list(np.arange(br, 1+br, br))
#        
#        cuts=df['X'].dropna().quantile(quant)
#        cuts=list(cuts)
#        
#        for i in cuts:
#            bins.append(i)
#
#        bins.append(float('Inf'))
#        self.bins=bins
        #cuts,
        breaks=pd.qcut(df["X"],self.breaks,retbins=True)[1]
        bins=[]
        bins.append(-float('Inf'))
        for i in breaks[1:self.breaks]:
            bins.append(i)
        #self.bins=bins
        bins.append(float('Inf'))
        self.bins=bins
        #self.bins = np.append((-float("inf"),), bins[1:-1])
        #self.bins=np.append(float('Inf'))
        #cuts, bins = pd.qcut(df["X"],16, retbins=True, labels=False)
        q = pd.cut(df['X'], bins=self.bins)
        df['labels']=q.astype(str)
        col_names = {'count_nonzero': 'bad', 'size': 'obs'}
        self.stat = df.groupby("labels")['Y'].agg([np.mean, np.count_nonzero, np.size]).rename(columns=col_names)   
        self.stat['bad_perc']=self.stat['bad']/sum(self.stat['bad'])
        self.stat['good']=self.stat['obs']-self.stat['bad']
        self.stat['good_perc']=self.stat['good']/sum(self.stat['good'])
        self.stat['woe'] = np.log(self.stat['good_perc'].values/self.stat['bad_perc'].values)
        self.stat['iv']= (self.stat['good_perc']-self.stat['bad_perc'])*self.stat['woe']
        self.iv=sum(self.stat['iv'])
       # return quant
    
    def deploy(self,df):
        ''' Deploy of bins '''
        labels = pd.cut(df[self.name],bins=self.bins)
        return labels  


m = met()
m.fit(dataset['EstimatedSalary'],dataset['Purchased'])
m.stat
m.iv
m.bins

dataset['deploy']=m.deploy(datase



"""
Created on Sat Nov  3 10:13:22 2018

@author: pc
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class woe:
    def __init__(self,bins=None,nbreaks=10):
        self.bins=bins if bins is None else bins
        self.stat=None 
        self.iv=None
        self.nbreaks=nbreaks
        self.name=None
        self.woe=None
        self.df=None

    def fit(self,x,y):
        '''Fitting Information'''
        if not isinstance(x, pd.Series):
            x = pd.Series(x.compute())
        if not isinstance(y, pd.Series):
            y = pd.Series(y.compute())
        self.name=x.name
        df = pd.DataFrame({"X": x, "Y": y, 'order': np.arange(x.size)})
        
        if self.bins is None:
           breaks=pd.qcut(df["X"],self.nbreaks,duplicates='drop',retbins=True)[1]
           breaks=breaks[1:-1]
           bins=[]
           bins.append(-float('Inf'))
           for i in breaks:
               bins.append(i)
           bins.append(float('Inf'))
           self.bins=bins
           
        q = pd.cut(df['X'], bins=self.bins,
                   labels=np.arange(len(self.bins)-1).astype(str))
        df['labels']=q.astype(str)
        q = pd.cut(df['X'], bins=self.bins)
        df['range']=q.astype(str)
        col_names = {'count_nonzero': 'bad', 'size': 'obs'}
        self.stat = df.groupby(["labels","range"])['Y'].agg([np.mean, np.count_nonzero, np.size]).rename(columns=col_names).copy()  
        self.stat['bad_perc']=self.stat['bad']/sum(self.stat['bad'])
        self.stat['good']=self.stat['obs']-self.stat['bad']
        self.stat['good_perc']=self.stat['good']/sum(self.stat['good'])
        self.stat['woe'] = np.log(self.stat['good_perc'].values/self.stat['bad_perc'].values)
        self.stat['iv']= (self.stat['good_perc']-self.stat['bad_perc'])*self.stat['woe']
        self.iv=sum(self.stat['iv'])
        self.df=df
                
    
    def deploy(self,df):
        ''' Deploy of bins '''

        if not isinstance(df[self.name], pd.Series):
            x = pd.Series(df[self.name].compute())
            labels = pd.cut(x,bins=self.bins,
                        labels=self.stat['woe'].tolist())
        if  isinstance(df[self.name], pd.Series):
             labels = pd.cut(df[self.name],bins=self.bins,
                        labels=self.stat['woe'].tolist())
            
        return labels  
    
    def plot(self):
        self.stat['index'] = self.stat.index
        ''' Plot in function of bad rate'''
        return self.stat.plot(kind='bar',x='index',y='mean',color='blue')
    
    def optimize(self,depth=2,criterion='gini'):
        clf = DecisionTreeClassifier(criterion='gini',random_state=0,
                                     max_depth=depth)
        
        clf.fit(self.df['X'][:, None],self.df['Y'])
        breaks=clf.tree_.threshold[clf.tree_.threshold>-2]
        breaks=sorted(breaks)
        bins=[]
        bins.append(-float('Inf'))
        for i in breaks:
            bins.append(i)
        bins.append(float('Inf'))
        self.bins=bins
        self.fit(self.df['X'],self.df['Y'])

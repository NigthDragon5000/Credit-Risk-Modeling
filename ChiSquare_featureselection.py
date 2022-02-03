# -*- coding: utf-8 -*-
"""
Created on Sat May  4 13:19:51 2019

@author: JAIR
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
#from scipy.stats import chi2_contingency

class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)

        print(result)
        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX,alpha)
    
    def globalTest(self,label):
        for var in self.df.columns:
            if var==label:
                pass
            else:
                self.TestIndependence(colX=var,colY=label)
                print('p value of: ',self.p)


df = pd.pandas.read_csv("train_titanic.csv")

cT = ChiSquare(df)
cT.globalTest('Survived')

cT.TestIndependence('Pclass','Survived')
print(cT.dfObserved.T)
print(cT.dfExpected.T)




# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 10:37:52 2019

@author: jcondori
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_white
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import numpy as np
import statsmodels.api as sm
from datetime import datetime

    
#
#def get_month(x):
#    if x=='Ene':
#        y=1
#    elif x=='Feb':
#        y=2
#    elif x=='Mar':
#        y=3
#    elif x=='Abr':
#        y=4
#    elif x=='May':
#        y=5
#    elif x=='Jun':
#        y=6
#    elif x=='Jul':
#        y=7
#    elif x=='Ago':
#        y=8
#    elif x=='Sep':
#        y=9
#    elif x=='Oct':
#        y=10
#    elif x=='Nov':
#        y=11
#    elif x=='Dic':
#        y=12
#    else:
#        y=0
#    return y


def get_month(x):
    if x=='Jan':
        y=1
    elif x=='Feb':
        y=2
    elif x=='Mar':
        y=3
    elif x=='Apr':
        y=4
    elif x=='May':
        y=5
    elif x=='Jun':
        y=6
    elif x=='Jul':
        y=7
    elif x=='Aug':
        y=8
    elif x=='Sep':
        y=9
    elif x=='Oct':
        y=10
    elif x=='Nov':
        y=11
    elif x=='Dec':
        y=12
    else:
        y=0
    return y


def monthly_dummie(df,month_number):
    '''
    Inputs:
        df : Data Frame, must hace a column with Month variables in format Ene,Feb,Sep...
    Returns:
        A Data Frame with months in one hot encode format
    '''
    #df['month_number']=df.apply(lambda row: get_month(row['month']),axis=1)       
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(df[month_number])
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded) 
    df.reset_index(inplace=True)
    df=pd.concat([df,pd.DataFrame(onehot_encoded)],axis=1)
    df.set_index('index',inplace=True)
    #df.drop('month_number',axis=1)
    return df



def DFtest(variable,p_value=0.05):
    result = adfuller(variable)
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
    if result[1] <= p_value:
        print('Conclusion : Stationary')
    else:
        print('Conclusion : Non stationary')



def error_analisis(result,plot=False):
    '''
    Inputs:
        result: Results from Stats after model.fit()
        plot: True if we want a plot
    Returns:
        Print of an statistics analysis of regression errors which includes: Autocorrelation, \
        Heterokedasticity, Stationarity and Normality
    '''
    #Autocorrleation
    print('----------Durbin Watson-------------')
    out = durbin_watson(result.resid)
    print('Durbin Watson is: '+str(out))

    if plot:
        qqplot(result.resid, line='s')
        pyplot.show()
    
    print('--------Breusch Autocorr-----------')

    try:
        bre=acorr_breusch_godfrey(result,nlags=12)
    
        print('lm: '+str(bre[0]))
        print('lmpval: '+str(bre[1]))
        print('fval: '+str(bre[2]))
        print('fpval: '+str(bre[3]))
    
        if bre[1] < 0.05:
            print('Evidence for autocorrelation')
        else:
            print('Not Evidence for autocorrelation')
    except:
        print('Cant calculate statistic')
        
    
    print('-----White Heteroskedasticity------')
    
    white_test = het_white(result.resid,  result.model.exog)
    
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    print(dict(zip(labels, white_test)))
    
    if white_test[1] < 0.05:
        print('Evidence for heteroskedasticity')
    else:
        print('Not Evidence for heteroskedasticity')
    
    
    print('----------ADF Test-----------------')
    try:
        DFtest(result.resid)
    except:
        print("Can't calculate ADF test")
    
    
    print('----------Shapiro Normality--------')
    stat, p = shapiro(result.resid)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
    	print('Sample looks Gaussian (fail to reject H0)')
    else:
    	print('Sample does not look Gaussian (reject H0)')
    
    if plot:
        residuals = pd.DataFrame(result.resid)
        plt.show()
        residuals.plot(kind='kde')
        plt.show()
            


def tm_plot(train,test,pred_train,pred_test,lags=24):
    plt.figure()
    plt.plot(train[-lags:], label='training')
    plt.plot(test, label='test')
    plt.plot(pred_train[-lags:], label='forecast in sample')
    plt.plot(pred_test, label='forecast')
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
    
    
def ac(result,lags=12,alpha=0.05):
    pacf = smt.graphics.plot_pacf(result.resid, lags=lags , alpha=alpha)
    pacf.show()
    
    acf = smt.graphics.plot_acf(result.resid, lags=lags , alpha=alpha)
    acf.show()


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    #acf1 = acf(fc-test)[1]                      # ACF1
    return({'1-mape':1-mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse,# 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})



def back_elimination(Y, X, alpha=0.05,frame=False,test=False,dftest=None,kind='robust'):
    numVars = len(X.columns)
    for i in range(0, numVars):
        X = sm.add_constant(X)
        if kind=='robust':
            regressor = sm.OLS(Y, X).fit(cov_type='HC3')
        elif kind=='no_robust':
            regressor = sm.OLS(Y, X).fit()
            
        if frame:
            print(regressor.summary())
        maxVar = max(regressor.pvalues[1:])#.astype(float)
        if maxVar > alpha:
            for name in regressor.pvalues.index:
                if (regressor.pvalues[name].astype(float) == maxVar) and name!='const': #\
               # and name!='const':
                    X=X.drop([name],axis=1)
                    if test:
                        dftest=dftest.drop([name],axis=1)
    print(regressor.summary())
    return regressor,X
    


def month_list(first_month,end_month):
    dates = [first_month, end_month]
    start, end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
    total_months = lambda dt: dt.month + 12 * dt.year
    mlist = []
    for tot_m in range(total_months(start)-1, total_months(end)):
        y, m = divmod(tot_m, 12)
        mlist.append(datetime(y, m+1, 1).strftime("%b-%y"))
    return mlist

def month(df):
    month=df['months'].str[:3]
    return month

def encode_month(df):
    df['month']=month(df)
    df=monthly_dummie(df)
    return df

def add_constant(df):
    df['const']=1
    return df

    

def rezagar(df,n=2,diferencias=False):
    
    if diferencias :
        for variable in  df.columns:
            df[variable+'_diff']=df[variable].diff()
    else:
        pass

    for variable in df.columns:
        for i  in range(1,n):
            df['_'+str(i)+str(variable)]=df[variable].shift(i)
    return df





# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:01:30 2019

@author: jcondori
"""

from pandas.io.json import json_normalize
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
#import pyodbc

class DFConverter:

    #Converts the input JSON to a DataFrame
    def convertToDF(self,dfJSON):
        return(json_normalize(dfJSON))

    #Converts the input DataFrame to JSON 
    def convertToJSON(self, df):
        resultJSON = df.to_json(orient='records')
        return(resultJSON)


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

''' Get al api del BCRP'''

data = requests.get("https://estadisticas.bcrp.gob.pe/estadisticas/series/api/PN01457BM-PN01652XM-PN01713AM-PN01714AM-PN01715AM-PN01716AM-PN01717AM-PN01718AM-PN01719AM-PN01720AM-PN01721AM-PN01722AM-PN01723AM-PN01724AM-PN01725AM-PN01728AM-PN01731AM-PD04722MM-PN07807NM-PN07816NM-PN01273PM-PN01711BM-PN02196PM-PN01234PM-PN37698XM/json/2009-1/2019-12",\
                    verify=False).json()

''' Extrayendo a panda'''

conv=DFConverter()
df=conv.convertToDF(data['periods'])
df=pd.DataFrame(df['values'].tolist())

names=conv.convertToDF(data['config']['series'])
df.columns=[names['name'].tolist()]

len(names['name'].tolist())



df.columns=['tasa_ref','tc','var_ipc','balanza_comercial','precio_cobre',\
            'var_ter_inter',\
            'pbi_agro','pbi_agro-agri','pbi_agro_pecu','pbi_pesca','pbi_min',\
            'pbi_min_meta','pbi_min_carbu','pbi_manu','pbi_manu_repri',\
            'pbi_manu_nopri', 'pbi_elec','pbi_constr','pbi_comer','pbi',\
            'pbi_des','ind_dempleo','TAMN','TIPMN','letras_10']

fechas=conv.convertToDF(data['periods'])
fechas=fechas['name'].tolist()

df.index=fechas

df.replace({'n.d.': np.nan}, inplace=True)

df.drop('letras_10',axis=1,inplace=True)

df=df.dropna()



''' Extrayendo de CSV'''

df2=pd.read_excel('crecimientoPBI.xlsx')

df=df.join(df2)

''' Transformando todo a numerico'''

cols = df.columns[df.dtypes.eq('object')]

df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')


''' Estadisticas'''

View=df.describe(include = 'all')

''' Transformaciones '''

df['SaldoVigMN_d']=df['SaldoVigMN'].diff()

df['var_per_sal']=df['SaldoVigMN'].pct_change(12)

df['pbi_d']=df['pbi'].diff()

df['tc_d']=df['tc'].diff()

df['var_per_analistas']=df['n_analistas'].pct_change(1)

df.dropna(inplace=True)

''' Transformacion '''

DFtest(df['pbi'])

DFtest(df['pbi_d'])

DFtest(df['var_per_sal'])

DFtest(df['tc_d'])

DFtest(df['n_analistas'])



''' Definiendo Espacio Muestral'''


df.reset_index(inplace=True)

df['anno'] = df['index'].str[4:]

df['anno']=df['anno'].astype('float')

test=df[df['anno']==2019]

train=df[df['anno']!=2019]

#df=df[df['anno']!='2010']

df=df.set_index('index')

train=train.set_index('index')

test=test.set_index('index')


#df=df[(df['level_0']>=5) & (df['level_0']<=90)] 

''' Grafico de ACF PACF '''

#fig = plt.figure(figsize=(12,8))
#ax1 = fig.add_subplot(211)
#fig = sm.graphics.tsa.plot_acf(train['var_per_sal'].dropna(), lags=20, ax=ax1)
#ax2 = fig.add_subplot(212)
#fig = sm.graphics.tsa.plot_pacf(train['var_per_sal'].dropna(), lags=20, ax=ax2)
#plt.show()


''' Probando Diferentes Lags de PBIs'''

for j in range(1,7):
    train2=train.copy()
    train2['const']=1
    train2['pbi'+str(j)]=train2['pbi'].shift(j)
    train2['pbi_diff_'+str(j)]=train2['pbi'+str(j)].diff()
    train2.dropna(inplace=True)
    model1=sm.OLS(endog=train2['var_per_sal'],exog=train2[['pbi_diff_'+str(j)]])
    results1=model1.fit()
    print(results1.summary())

''' Auto ARIMA'''

from pyramid.arima import auto_arima

#train = train[['var_per_sal']]

#model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
#res=model.fit(train)
#model.summary()


train['pbi_1']=train['pbi'].shift(1)
train['pbi_2']=train['pbi'].shift(2)
train['pbi_3']=train['pbi'].shift(3)
train['pbi_4']=train['pbi'].shift(4)
train['pbi_5']=train['pbi'].shift(5)
train['pbi_6']=train['pbi'].shift(6)
train.dropna(inplace=True)

# Por default se ajusta por AIC 
model = auto_arima(train[['var_per_sal']], exogenous=train[[
                    'pbi',
                   #'pbi_1',
                   #'pbi_2',\
                   #'pbi_3',
                   #'pbi_4',
                   'pbi_5'
                   #,'pbi_6'
                   ]],
                      start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=12, max_q=12, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

results = model

print(model.summary())



''' ARIMA'''


#df['lag']=df['diffM'].shift()
#df.dropna(inplace=True)
model=sm.tsa.ARIMA(endog=train['var_per_sal'],exog=train[['pbi_6']],\
                    order=[1,1,0])
results=model.fit(disp=-1)
print(results.summary())

#results3.predict(df[['var_per_sal','pbi_d','const']])

''' Revisando lo ajustado'''

predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
#predictions_ARIMA = pd.Series(df['var_per_sal'].iloc[0], index=df.index)
predictions_ARIMA_diff=pd.DataFrame(predictions_ARIMA_diff)
predictions_ARIMA_diff.columns=['ARIMA']

train=train.merge(predictions_ARIMA_diff_cumsum.to_frame(), how='left',left_index=True, right_index=True)
train['ARIMA']=train['var_per_sal'].iloc[0]
train['ARIMA']=train['ARIMA']+train[0]
train=train.drop(0,axis=1)

#train['ARIMA']=predictions_ARIMA_diff

plt.show()
plt.plot(train['var_per_sal'])
plt.plot(train['ARIMA'])
plt.show()


''' R2 '''

from math import pow

print('R2: '+str(pow(train[['var_per_sal','ARIMA']].corr().iloc[0,1],2)))

''' Preparando para prediccion '''

#model.predict()
#yhat = results.predict(len(df), len(df)+1, exog=[-2,2])
#yhat = results.predict(len(df), len(df)+2, exog=[-2,2,3])

#exogenas=pd.DataFrame(df[['pbi','tc']])

exogenas=pd.DataFrame(df[['pbi']])

exogenas_proyectado=pd.DataFrame()


fechas_proyectado=['Ene.2019','Feb.2019','Mar.2019','Abr.2019','May.2019','Jun.2019','Jul.2019','Ago.2019','Sep.2019','Oct.2019','Nov.2019','Dic.2019',\
                   'Ene.2020','Feb.2020','Mar.2020','Abr.2020','May.2020','Jun.2020','Jul.2020','Ago.2020','Sep.2020','Oct.2020','Nov.2020','Dic.2020',\
                   'Ene.2021','Feb.2021','Mar.2021','Abr.2021','May.2021','Jun.2021','Jul.2021','Ago.2021','Sep.2021','Oct.2021','Nov.2021','Dic.2021']

# Tomamos fechas proyectadas desde Junio
exogenas_proyectado = exogenas_proyectado.reindex(fechas_proyectado[5:])

exogenas_proyectado2=exogenas_proyectado.copy()
#
exogenas_proyectado['pbi']=[3,3,3,3,3,3,3,\
                2.4,2.4,2.4,2.4,2.4,2.4,2.4,2.4,2.4,2.4,2.4,2.4,\
                1.9,1.9,1.9,1.9,1.9,1.9,1.9,1.9,1.9,1.9,1.9,1.9]
#
exogenas_proyectado2['pbi']=[0,0,0,0,0,0,0,\
                0,0,0,0,0,0,0,0,0,0,0,0,\
                0,0,0,0,0,0,0,0,0,0,0,0]

#
#exogenas_proyectado['tc']=[3.3,3.3,3.3,3.3,3.3,3.3,3.3,\
#                3.3,3.3,3.3,3.3,3.3,3.3,3.3,3.3,3.3,3.3,3.3,3.3,\
#                3.3,3.3,3.3,3.3,3.3,3.3,3.3,3.3,3.3,3.3,3.3,3.3]


#tc = []
#for j in range(0,31):
#    tc.append(3.4)
#    
#
#exogenas_proyectado2['tc']=tc
#
#exogenas_proyectado2['tc']=[3.5,3.5,3.5,3.5,3.5,3.5,3.5,\
#                3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5,\
#                3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5]


exogenas1=exogenas.append(exogenas_proyectado)

exogenas1['pbi_d']=exogenas1['pbi'].diff()
#
#exogenas1['tc_d']=exogenas1['tc'].diff()


exogenas2=exogenas.append(exogenas_proyectado2)

exogenas2['pbi_d']=exogenas2['pbi'].diff()

#exogenas2['tc_d']=exogenas2['tc'].diff()

#exog=pbi['pbi_d'][-36:]


''' Predicciones'''

def forecast(steps,exogenas):
    #steps=15
    
    if steps==36:
        exog=exogenas[['pbi_d']]
    else:
        exog=exogenas[['pbi_d']][-36:-36+steps]
    
    
    fc, se, conf = results.forecast(steps=steps, exog=exog\
                                    ,alpha=0.05)  # 95% conf
    
    fc_series = pd.Series(fc, index=fechas_proyectado[0:steps])
    lower_series = pd.Series(conf[:, 0], index=fechas_proyectado[0:steps])
    upper_series = pd.Series(conf[:, 1], index=fechas_proyectado[0:steps])
    
    return fc_series,lower_series,upper_series

fc_series,lower_series,upper_series=forecast(35,exogenas1)
fc_series1,lower_series1,upper_series1=forecast(35,exogenas2)

fc,l,u=forecast(5,exogenas1)


''' Grafico Predicciones '''


plt.figure()
#plt.figure(figsize=(12,5), dpi=100)
plt.plot(train['var_per_sal'][-6:], label='training')
plt.plot(test['var_per_sal'], label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='g', alpha=.15)
plt.plot(fc_series1, label='forecast2')
plt.fill_between(lower_series1.index, lower_series1, upper_series1, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

''' Accuracy metrics'''

#from statsmodels.tsa.stattools import acf

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

forecast_accuracy(fc, test['var_per_sal'])


''' VAR '''


from statsmodels.tsa.api import VAR

#train_var=train[['var_per_sal','pbi','n_analistas']]

train_var=train[['var_per_sal','pbi']]

model = VAR(train_var)
for i in [1,2,3,4,5,6,7,8,9]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')
    
    
model_fitted = model.fit(2)
model_fitted.summary()

''' Autocorrelacion '''

from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(model_fitted.resid)

#results.plot_diagnostics(figsize=(7,5))
#plt.show()
#

#print(residuals.describe())
#
#DFtest(residuals[0].tolist())

''' Cointegracion ? '''

DFtest(result.resid['var_per_sal'])


''' Normality  '''

residuals = pd.DataFrame(results.resid)
#residuals.plot()
#plt.show()
residuals.plot(kind='kde')
plt.show()

#from numpy.random import randn
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
# seed the random number generator
# generate univariate observations
# q-q plot
qqplot(result.resid['var_per_sal'], line='s')
pyplot.show()


from scipy.stats import shapiro

stat, p = shapiro(result.resid['var_per_sal'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')



''' Prediccion '''


lag_order = model_fitted.k_ar
lag_order =  2
print(lag_order)  #> 4

# Input data for forecasting
forecast_input = train_var[-lag_order:]
forecast_input = forecast_input.values

nobs=30
fc = model_fitted.forecast(y=forecast_input, steps=nobs )
df_forecast = pd.DataFrame(fc, index=df.index[-nobs:], columns=df.columns + '_2d')
df_forecast


exportacion = df[['var_per_sal','pbi','n_analistas','SaldoVigMN']]
exportacion.to_excel('train_var.xlsx')



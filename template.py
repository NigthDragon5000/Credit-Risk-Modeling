# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

import pandas as pd
#import numpy as np 
import os
import pyodbc
import gc
import dask.dataframe as dd

''' using Garbage Collector'''
gc.collect()

#Cambiando Directorio
os.chdir("D:/")


'''' Extracting Data with Dask'''
#df = dd.read_csv("salidaSobreV2.csv",
#                 dtype={'END_ENTPROM_SALDO_TOTAL_U12M': 'float64',
#       'END_ENTPROM_SALDO_TOTAL_U6M': 'float64'})

df = dd.read_csv("GrupGrup4.csv",
                 dtype={'MAX_COMP_MES_ULT_PCALIF_U12M': 'float64',
       'MAX_COMP_MES_ULT_PCALIF_U12M_X': 'float64',
       'MAX_COMP_MES_ULT_PCALIF_U24M': 'float64',
       'MAX_COMP_MES_ULT_PCALIF_U24M_X': 'float64',
       'MAX_END_ENTPROM_SALDO_MICRO_U12M': 'float64',
       'MAX_END_ENTPROM_SALDO_MICRO_U6M': 'float64',
       'MAX_END_MAX_SOW_U12M': 'float64',
       'MAX_END_MAX_SOW_U3M': 'float64',
       'MAX_END_MAX_SOW_U6M': 'float64',
       'MAX_END_MAX_SOW_U9M': 'float64',
       'MAX_END_PROM_SOW_U12M': 'float64',
       'MAX_END_PROM_SOW_U3M': 'float64',
       'MAX_END_PROM_SOW_U6M': 'float64',
       'MAX_END_PROM_SOW_U9M': 'float64',
       'MAX_END_SALDOMAXCOMP_SALDOMAX_MICRO_U12M': 'float64',
       'MAX_END_SALDOMAXCOMP_SALDOMAX_MICRO_U24M': 'float64',
       'MAX_END_SALDOMAXCOMP_SALDOMAX_MICRO_U6M': 'float64',
       'MAX_END_SALDOMAXCOMP_SALDOMAX_MICRO_U9M': 'float64',
       'MAX_END_SALDOMAXCOMP_SALDOMAX_TOTAL_U12M': 'float64',
       'MAX_END_SALDOMAXCOMP_SALDOMAX_TOTAL_U24M': 'float64',
       'MAX_END_SALDOMAXCOMP_SALDOMAX_TOTAL_U6M': 'float64',
       'MAX_END_SALDOMAXCOMP_SALDOMAX_TOTAL_U9M': 'float64',
       'MAX_END_ULTIMO_SOW': 'float64',
       'MAX_EXP_CANT_MES_PRIM_REF_MICRO_U12M': 'float64',
       'MAX_EXP_CANT_MES_PRIM_REF_MICRO_U18M': 'float64',
       'MAX_EXP_CANT_MES_PRIM_REF_MICRO_U24M': 'float64',
       'MAX_EXP_CANT_MES_PRIM_REF_MICRO_U36M': 'float64',
       'MAX_EXP_CANT_MES_PRIM_REF_MICRO_U48M': 'float64',
       'MAX_EXP_CANT_MES_PRIM_REF_MICRO_U60M': 'float64',
       'MAX_EXP_CRED_MICRO_U12M': 'float64',
       'MAX_EXP_CRED_MICRO_U24M': 'float64',
       'MAX_EXP_CRED_MICRO_U6M': 'float64',
       'MAX_EXP_MES_ULT_REP_COMP_U12M': 'float64',
       'MAX_EXP_MES_ULT_REP_COMP_U24M': 'float64',
       'MAX_EXP_NUM_MES_ULT_OBLI_MICRO_U12M': 'float64',
       'MAX_EXP_NUM_MES_ULT_OBLI_MICRO_U18M': 'float64',
       'MAX_EXP_NUM_MES_ULT_OBLI_MICRO_U24M': 'float64',
       'MAX_EXP_NUM_MES_ULT_OBLI_MICRO_U36M': 'float64',
       'MAX_EXP_NUM_MES_ULT_OBLI_MICRO_U48M': 'float64',
       'MAX_EXP_NUM_MES_ULT_OBLI_MICRO_U60M': 'float64',
       'ciclo_gru': 'float64',
       'nAtrMax_gru': 'float64',
       'MAX_END_ENTPROM_SALDO_TOTAL_U12M': 'float64',
       'MAX_END_ENTPROM_SALDO_TOTAL_U6M': 'float64',
       'MAX_EXP_CANT_MES_PRIM_REF_REP_U12M': 'float64',
       'MAX_EXP_CANT_MES_PRIM_REF_REP_U18M': 'float64',
       'MAX_EXP_CANT_MES_PRIM_REF_REP_U24M': 'float64',
       'MAX_EXP_CANT_MES_PRIM_REF_TOTAL_U12M': 'float64',
       'MAX_EXP_CANT_MES_PRIM_REF_TOTAL_U18M': 'float64',
       'MAX_EXP_CANT_MES_PRIM_REF_TOTAL_U24M': 'float64',
       'MAX_EXP_NUM_MES_ULT_OBLI_REP_U12M': 'float64',
       'MAX_EXP_NUM_MES_ULT_OBLI_REP_U18M': 'float64',
       'MAX_EXP_NUM_MES_ULT_OBLI_REP_U24M': 'float64',
       'MAX_EXP_NUM_MES_ULT_OBLI_TOTAL_U12M': 'float64',
       'MAX_EXP_NUM_MES_ULT_OBLI_TOTAL_U18M': 'float64',
       'MAX_EXP_NUM_MES_ULT_OBLI_TOTAL_U24M': 'float64'})

#len(df)

df.npartitions

''' Partition Data with Dask'''
#from dklearn.cross_validation import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(df,test_size=0)

# Se cambio la definicion de buenso a <=7 
base=df[(df.id0==0) &  (df.id6!=2)]

#base=df[(df.atraso0_cronograma<=7) &  (df.id6!=2)]

''' Reemplazando data donde es necesario'''

#df.assign(id6inv = lambda x: x.id6-1) 

#result = df.id6.where(df.id6 == 1, 0) 

#df = df.mask(df['id6'] == 0, 1).mask(df['id6'] == 1, 0)


base['id6_inv']=base['id6']

def replace(x: pd.DataFrame) -> pd.DataFrame:
    return x.replace(
      {'id6_inv': [0, 1]},
      {'id6_inv': [1, 0]}
    )
base = base.map_partitions(replace)

View= base[['id6','ccodcta_gru','dfecrep','id6_inv']].head()

''' Train Test Split'''

train, test = base.random_split([0.80, 0.20],random_state = 123)

#len(train)
#len(test)
df.columns

''' Extrayendo'''

train=train.compute()
test=test.compute()


#dask_ml.model_selection.train_test_split(df7, random_state=0)
''' Check Nulls y Anomalias'''

sum(train['AVG_END_PROM_SOW_U3M'].isnull())

train = train[train['AVG_END_PROM_SOW_U3M'].notnull()]
test = test[test['AVG_END_PROM_SOW_U3M'].notnull()]

#34392 34391
#136423 136420


print(train.describe())


from woe5 import woe

''' Check Categoricals'''

pru=train[['categoria_analista','id6']]


cate = ['JUNIOR                                                                '
        ,'MASTER                                                                ',
        'SENIOR                                                                ']

pru=pru[pru.categoria_analista.isin(cate)]

w=woe()
w.fit_categorical(pru['categoria_analista'],pru['id6'])
framecategoria=w.stat
# iv menor al 1%



'''' Woes Massive '''

train_eval=train.iloc[:,0:500]

frames=[]
names=[]
iv=[]
monotonic=[]
lista=[]
per_NA = []
depth_arbol=[]
#for i in train.iloc[:,0:20].columns.tolist():
for i in train_eval.columns.tolist():
   try :
      w=woe(nbreaks=10)
      w.fit(train[i],train['id6'])
      for j in list(range(1,4)):
          w.optimize(depth=j,samples=int(round(len(train)*0.05)),max_nodes=5,seed=0) # Minimo por leaf min_sample_leaf
          if w._checkMonotonic() and w.iv != float('Inf') and w.iv >0.02:
              frames.append(w.stat)
              names.append(i)
              iv.append(w.iv)
              monotonic.append(w._checkMonotonic())
              per_NA.append(w.per_NA)
              depth_arbol.append(j)
#              train[str(i+'_binned')]=w.deploy(train)
#              test[str(i+'_binned')]=w.deploy(test)              
   except KeyboardInterrupt:
       raise Exception('Stop by user')
   except:
        pass

dm =  pd.DataFrame({'Names':names, 'IV':iv, 'Monotono' : monotonic\
                    ,'per_NA': per_NA,'depth':depth_arbol})


g= dm.groupby(['Names']) 
g['IV'].max()    
    
#frames_backup=frames.copy()
#dm_backup=dm.copy()

df.filter(like='result',axis=1)

''' Regresando a dask'''
train= dd.from_pandas(train,npartitions=18)
test = dd.from_pandas(test,npartitions=18)
df2 = dd.from_pandas(df2,npartitions=18)

vista=frames[1][['bad','bad_perc','breaks','good','good_perc','iv',	
      'mean','obs','per','woe']]



'''Woes Individuales'''

#df2 = df.reset_index().set_index('index').copy() 
df2=df[df['AVG_END_MAX_SOW_U3M'].notnull()].compute() # 4 casos

# atraso0_cronograma
w_atraso0_cronograma=woe(bins=[-float('Inf'),0,1,float('Inf')])
w_atraso0_cronograma.fit(train['atraso0_cronograma'],train['id6'])
frame=w_atraso0_cronograma.stat[['z','breaks','good','good_perc','bad','bad_perc','mean','iv','per','woe']]
train['atraso0_cronograma_binned']=w_atraso0_cronograma.deploy(train)
test['atraso0_cronograma_binned']=w_atraso0_cronograma.deploy(test)  
df2['atraso0_cronograma_binned']=w_atraso0_cronograma.deploy(df2)  



# AVG_natrMax
w_AVG_natrMax=woe(bins=[-float('Inf'),0.22,0.38,0.50,0.87,float('Inf')])
w_AVG_natrMax.fit(train['AVG_natrMax'],train['id6'])
frame=w_AVG_natrMax.stat[['z','breaks','good','good_perc','bad','bad_perc','mean','iv','per','woe']]
train['AVG_natrMax_binned']=w_AVG_natrMax.deploy(train)
test['AVG_natrMax_binned']=w_AVG_natrMax.deploy(test)  
df2['AVG_natrMax_binned']=w_AVG_natrMax.deploy(df2)  



# nAtrmax_gru
w_nAtrMax_gru=woe(bins=[-float('Inf'),3,4,11,14,float('Inf')])
w_nAtrMax_gru.fit(train['nAtrMax_gru'],train['id6'])
frame=w_nAtrMax_gru.stat[['z','breaks','good','good_perc','bad','bad_perc','mean','iv','per','woe']]
train['nAtrMax_gru_binned']=w_nAtrMax_gru.deploy(train)
test['nAtrMax_gru_binned']=w_nAtrMax_gru.deploy(test)  
df2['nAtrMax_gru_binned']=w_nAtrMax_gru.deploy(df2)  



# AVG_ccal_cli
w_AVG_ccal_cli=woe(bins=[-float('Inf'),0.57,1.07,1.66,2.00,float('Inf')])
w_AVG_ccal_cli.fit(train['AVG_ccal_cli'],train['id6'])
frame=w_AVG_ccal_cli.stat[['z','breaks','good','good_perc','bad','bad_perc','mean','iv','per','woe']]
train['AVG_ccal_cli_binned']=w_AVG_ccal_cli.deploy(train)
test['AVG_ccal_cli_binned']=w_AVG_ccal_cli.deploy(test)  
df2['AVG_ccal_cli_binned']=w_AVG_ccal_cli.deploy(df2)  


# COMP_ATR_CLI_GRU_CONTABLE
w_COMP_ATR_CLI_GRU_CONTABLE=woe(bins=[-float('Inf'),-2,float('Inf')])
w_COMP_ATR_CLI_GRU_CONTABLE.fit(train['COMP_ATR_CLI_GRU_CONTABLE'],train['id6'])
frame=w_COMP_ATR_CLI_GRU_CONTABLE.stat[['z','breaks','good','good_perc','bad','bad_perc','mean','iv','per','woe']]
train['COMP_ATR_CLI_GRU_CONTABLE_binned']=w_COMP_ATR_CLI_GRU_CONTABLE.deploy(train)
test['COMP_ATR_CLI_GRU_CONTABLE_binned']=w_COMP_ATR_CLI_GRU_CONTABLE.deploy(test)  
df2['COMP_ATR_CLI_GRU_CONTABLE_binned']=w_COMP_ATR_CLI_GRU_CONTABLE.deploy(df2)  



# AVG_END_MAX_SOW_U3M
w_AVG_END_MAX_SOW_U3M=woe(bins=[-float('Inf'),0.38,0.48,0.56,0.63,float('Inf')])
w_AVG_END_MAX_SOW_U3M.fit(train['AVG_END_MAX_SOW_U3M'],train['id6'])
frame=w_AVG_END_MAX_SOW_U3M.stat[['z','breaks','good','good_perc','bad','bad_perc','mean','iv','per','woe']]
train['AVG_END_MAX_SOW_U3M_binned']=w_AVG_END_MAX_SOW_U3M.deploy(train)
test['AVG_END_MAX_SOW_U3M_binned']=w_AVG_END_MAX_SOW_U3M.deploy(test)  
df2['AVG_END_MAX_SOW_U3M_binned']=w_AVG_END_MAX_SOW_U3M.deploy(df2)  




#AVG_EXP_NUM_MES_TOTAL_U12M
w_AVG_EXP_NUM_MES_TOTAL_U12M=woe(bins=[-float('Inf'),6.13,9.55,10.87,11.32,float('Inf')])
w_AVG_EXP_NUM_MES_TOTAL_U12M.fit(train['AVG_EXP_NUM_MES_TOTAL_U12M'],train['id6'])
frame=w_AVG_EXP_NUM_MES_TOTAL_U12M.stat[['z','breaks','good','good_perc','bad','bad_perc','mean','iv','per','woe']]
train['AVG_EXP_NUM_MES_TOTAL_U12M_binned']=w_AVG_EXP_NUM_MES_TOTAL_U12M.deploy(train)
test['AVG_EXP_NUM_MES_TOTAL_U12M_binned']=w_AVG_EXP_NUM_MES_TOTAL_U12M.deploy(test)  
df2['AVG_EXP_NUM_MES_TOTAL_U12M_binned']=w_AVG_EXP_NUM_MES_TOTAL_U12M.deploy(df2)  

# probando clientes promedio del grupo




''' Uniendo bins con NA'''
ww=woe()
ww=woe(bins=[-float('Inf'),0.4919,float('Inf')])
ww.fit(train['per_mal_calificados'],train['id6'])
ww.optimize(depth=1,samples=int(round(len(train)*0.05)),max_nodes=5,seed=0)
qq=ww.stat


#frames[16]=w.stat
obj=w.stat
obj=w.merge(obj,0,5)

#obj=obj[['z','breaks','good','good_perc','bad','bad_perc','mean','iv','per','woe']]


train['AVG_EXP_CANT_MES_PRIM_REF_MICRO_U12M_binned_NA']=w.deploy_frame(obj,train)
test['AVG_EXP_CANT_MES_PRIM_REF_MICRO_U12M_binned_NA']=w.deploy_frame(obj,test)
df2['AVG_EXP_CANT_MES_PRIM_REF_MICRO_U12M_binned_NA']=w.deploy_frame(obj,df2)



#ciclo _ gru
w_ciclo_gru=woe(bins=[-float('Inf'),1,2,4,11,float('Inf')])
w_ciclo_gru.fit(train['ciclo_gru'],train['id6'])
frame=w_ciclo_gru.stat[['z','breaks','good','good_perc','bad','bad_perc','mean','iv','per','woe']]
train['ciclo_gru_binned']=w_ciclo_gru.deploy(train)
test['ciclo_gru_binned']=w_ciclo_gru.deploy(test)  
df2['ciclo_gru_binned']=w_ciclo_gru.deploy(df2)  


#0.41421568393707275
#0.49193547666072845
#inf


#Porcentaje de mal calificados 
w_per_mal_calificados=woe()
w_per_mal_calificados.fit(train['per_mal_calificados'],train['id6'])
w_per_mal_calificados.optimize(depth=1,samples=int(round(len(train)*0.05)),max_nodes=5,seed=0)
frame=w_per_mal_calificados.stat[['z','breaks','good','good_perc','bad','bad_perc','mean','iv','per','woe']]
train['per_mal_calificados_binned']=w_per_mal_calificados.deploy(train)
test['per_mal_calificados_binned']=w_per_mal_calificados.deploy(test)  
df2['per_mal_calificados_binned']=w_per_mal_calificados.deploy(df2)  



#df2['AVG_EXP_CANT_MES_PRIM_REF_MICRO_U12M_binned_NA']=w.deploy_frame(df2)

#framee=train[['atraso0_cronograma','id6']]
#obj2=frames[2]
#framee['atrasitos_binned']=w.deploy_frame(obj2,framee)


''' Guardando '''
#train_saved=train
#test_saved=test
#dm_saved=dm

''' Seleccionando Data para stats model'''


x_train=train[
        [
'atraso0_cronograma_binned',
'AVG_natrMax_binned',
#'nAtrMax_gru_binned'  ,
# Menores a un IV de 20%
'AVG_ccal_cli_binned',
'COMP_ATR_CLI_GRU_CONTABLE_binned',
#'SUM_END_ENT_SALDO_TOTAL_UM_binned',
#'SUM_END_ENT_REP_TOTAL_UM_binned',
#'AVG_END_NUM_INCREM_SALDO_MICRO_U6M_binned',
#'END_NUM_CLI_GRU_binned' ,
#IV menores de 15%
'AVG_END_MAX_SOW_U3M_binned',
'AVG_EXP_NUM_MES_TOTAL_U12M_binned',
#'AVG_EXP_CANT_MES_PRIM_REF_MICRO_U12M_binned_NA',
# IV entre 5 y 13%
#'AVG_nexpcli_binned',
'ciclo_gru_binned',
#'AVG_edad_binned',
#'AVG_END_DIF_LINEA_TC_U12M_binned'
#'per_mal_calificados_binned'
                ]].compute()


y_train=train['id6_inv'].compute()



#test=test[test['atraso0_cronograma']<=0]




x_test=test[
        [
'atraso0_cronograma_binned',
'AVG_natrMax_binned',
#'nAtrMax_gru_binned'  ,
# Menores a un IV de 20%
'AVG_ccal_cli_binned',
'COMP_ATR_CLI_GRU_CONTABLE_binned',
#'SUM_END_ENT_SALDO_TOTAL_UM_binned',
#'SUM_END_ENT_REP_TOTAL_UM_binned',
#'AVG_END_NUM_INCREM_SALDO_MICRO_U6M_binned',
#'END_NUM_CLI_GRU_binned' ,
#IV menores de 15%
'AVG_END_MAX_SOW_U3M_binned',
'AVG_EXP_NUM_MES_TOTAL_U12M_binned',
#'AVG_EXP_CANT_MES_PRIM_REF_MICRO_U12M_binned_NA',
# IV entre 5 y 13%
#'AVG_nexpcli_binned',
'ciclo_gru_binned',
#'AVG_edad_binned',
#'AVG_END_DIF_LINEA_TC_U12M_binned'
                ]].compute()

y_test=test['id6_inv'].compute()


### Incluyendo Indeterminados


df2['id6_inv']=df2['id6']

def replace(x: pd.DataFrame) -> pd.DataFrame:
    return x.replace(
      {'id6_inv': [0, 1]},
      {'id6_inv': [1, 0]}
    )
df2 = df2.map_partitions(replace)


df2=df2.compute()




df2_base=df2[
        [
'atraso0_cronograma_binned',
'AVG_natrMax_binned',
#'nAtrMax_gru_binned'  ,
# Menores a un IV de 20%
'AVG_ccal_cli_binned',
'COMP_ATR_CLI_GRU_CONTABLE_binned',
#'SUM_END_ENT_SALDO_TOTAL_UM_binned',
#'SUM_END_ENT_REP_TOTAL_UM_binned',
#'AVG_END_NUM_INCREM_SALDO_MICRO_U6M_binned',
#'END_NUM_CLI_GRU_binned' ,
#IV menores de 15%
'AVG_END_MAX_SOW_U3M_binned',
'AVG_EXP_NUM_MES_TOTAL_U12M_binned',
#'AVG_EXP_CANT_MES_PRIM_REF_MICRO_U12M_binned_NA',
# IV entre 5 y 13%
#'AVG_nexpcli_binned',
'ciclo_gru_binned',
#'AVG_edad_binned',
#'AVG_END_DIF_LINEA_TC_U12M_binned'
                ]].compute()


df2_pred=df2[
        [
'ccodcta_gru','dfecrep','id0','id6','saldo_gru', 'saldo_gru6'
                ]].compute()



''' Verificacion de Correlacion y Vector de Inflacion de Varianza '''

corr=x_train.corr(method='pearson')
corr=df2_base.corr(method='pearson')


from statsmodels.stats.outliers_influence import variance_inflation_factor

base_corr = x_train.values


vif = [variance_inflation_factor(base_corr, i) for i in range(base_corr.shape[1])]
print(vif)


''' Chequeando Vacios'''

percent_missing = df.isnull().sum() * 100 / len(df)

#
#import statsmodels.api as sm
#
#y = x_train['const']
#X = x_train[['AVG_natrMax_binned','atraso0_cronograma_binned',
#       'nAtrMax_gru_binned', 'AVG_ccal_cli_binned',
#       'COMP_ATR_CLI_GRU_CONTABLE_binned', 'AVG_END_MAX_SOW_U3M_binned',
#       'AVG_EXP_CANT_MES_PRIM_REF_MICRO_U12M_binned_NA',
#       'AVG_EXP_NUM_MES_TOTAL_U12M_binned']]
#
## Note the difference in argument order
#model = sm.OLS(y, X).fit()
##predictions = model.predict(X) # make the predictions by the model
#
## Print out the statistics
#model.summary()


''' Modelo con stats Model'''

import statsmodels.api as sm
from scipy import stats

#x_train=x_train.fillna(0)
x_train = sm.add_constant(x_train)
est = sm.Logit(y_train, x_train)
res = est.fit()
print(res.summary())

''' beta estandarizada '''

z=zscore(x_train)
z[:, 0] =  1
z=pd.DataFrame(z)
z.columns=list(x_train.columns)
from scipy.stats.mstats import zscore
print (sm.Logit(y_train, z.values).fit().summary())
z.columns

''' Predicciones con statsmodel'''
#x_validation=x_validation.fillna(0)
x_test = sm.add_constant(x_test)
y_pred=res.predict(x_test)
y_pred_train=res.predict(x_train)


#df2_back=df2_base.copy()
df2_base = sm.add_constant(df2_base)
df2_pd=res.predict(df2_base)
df2_base=pd.concat([df2_base,df2_pd,df2_pred],axis=1)
df2_base['pd']=df2_base.iloc[:,8]

#k=df2_pred[['saldo_gru','saldo_gru6']]
#df2_base2=pd.merge(df2_base,k,left_on=['ccodcta_gru','dfecrep'],right_on=['ccodcta_gru','dfecrep'])
#df2_base2=df2_base2.reindex()

#df3 = pd.concat(objs=[df2_base,k],axis=1)

### Comprobacion 
df2_pre=df2_base[['id6','id0','pd']]
df2_pre  = df2_pre[(df2_pre.id6 != 2)   & (df2_pre.id0== 0)]
## & (df2_pre.id0== 0)
valor,table=ks(df2_pre['id6'].values,df2_pre.iloc[:,2].values)
#
#len(train)
#len(test)


''' AUC'''

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
logit_roc_auc_train = roc_auc_score(y_train, y_pred_train)
#fpr, tpr, thresholds = roc_curve(y_validation, logisticRegr.predict_proba(x_validation)[:,1])
fpr, tpr, thresholds = roc_curve(y_train, y_pred_train)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc_train)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic TRAIN')
plt.legend(loc="lower right")
plt.savefig('Log_ROC_train')
plt.show()

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
logit_roc_auc_test = roc_auc_score(y_test, y_pred)
#fpr, tpr, thresholds = roc_curve(y_validation, logisticRegr.predict_proba(x_validation)[:,1])
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc_test)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic TEST')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

print(logit_roc_auc_train)
print(logit_roc_auc_test)

''' KS Distance'''

#value_train, pvalue = stats.ks_2samp(
#        fpr, tpr)
#
#value_test, pvalue = stats.ks_2samp(
#   y_pred.values,y_test.values)

from tools import ks

valor_train,table_train=ks(y_train,y_pred_train)
valor_test,table_test=ks(y_test,y_pred)

print(valor_train,valor_test)

#export =  pd.DataFrame({'bad':y_test, 'score':y_pred})

#export.to_csv('export.csv', sep='\t')

''' Diccionario'''

dic =	{
 # names[1]: frames[1]
}


for i in list(range(len(names))):
    dic[names[i]] = [frames[i],iv[i]]
    
    
''' Score Card '''


variables=[ 'atraso0_cronograma','AVG_natrMax','nAtrMax_gru' ,'AVG_ccal_cli',
            'COMP_ATR_CLI_GRU_CONTABLE','SUM_END_ENT_SALDO_TOTAL_UM',
            'AVG_END_NUM_INCREM_SALDO_MICRO_U6M','AVG_END_MAX_SOW_U3M','AVG_EXP_NUM_MES_TOTAL_U12M',
            'AVG_EXP_CANT_MES_PRIM_REF_MICRO_U12M']

ScoreCard=None

for i in variables:
    #print(dic[i][0])
    dic[i][0]['Name']=i
    ScoreCard=pd.concat([ScoreCard,dic[i][0]],sort=False)

ScoreCard=ScoreCard[['Name','breaks','good','good_perc','bad','bad_perc','mean','iv','per','woe']]


''' Poniendo Puntajes'''

import math

df2_base['puntaje']= (res.params[0]+ \
df2_base['atraso0_cronograma_binned']*res.params[1]+ \
df2_base['AVG_natrMax_binned']*res.params[2]+ \
#df2_base['nAtrMax_gru_binned']*res.params[3]+ \
df2_base['AVG_ccal_cli_binned']*res.params[3]+ \
df2_base['COMP_ATR_CLI_GRU_CONTABLE_binned']*res.params[4]+ \
df2_base['AVG_END_MAX_SOW_U3M_binned']*res.params[5]+ \
df2_base['AVG_EXP_NUM_MES_TOTAL_U12M_binned']*res.params[6] + \
df2_base['ciclo_gru_binned']*res.params[7] )  *80/math.log(2)+ 600
#df2_base['AVG_EXP_CANT_MES_PRIM_REF_MICRO_U12M_binned_NA']*res.params[8]) *80/math.log(2)+\

        
        
''' Percentilizar'''

# Como dropear
#df2_base=df2_base.drop(['ccodcta_gru', 'dfecrep'], axis=1).copy()


#exportar=df2[['ccodcta_gru','dfecrep','id6','id0']].compute()

#df2_base=pd.concat([exportar,df2_base],axis=1)

## El campo bucket de df2_base debe variarse por la deficnion que se haga en la base desarrollo
#df2_base['bucket'] = pd.qcut(df2_base['puntaje'], 20 ,duplicates='drop',retbins=True)[0]

col_names = {'count_nonzero': 'tasamalos', 'size': 'obs'}
analisis1=df2_base[df2_base['id0']!=1]
analisis1['bucket'] = pd.qcut(analisis1['puntaje'], 20 ,duplicates='drop',retbins=True)[0]
print(analisis1[analisis1['id6']!=2].groupby('bucket')['id6']\
.agg([np.mean, np.count_nonzero, np.size]).rename(columns=col_names))

analisis1.groupby('bucket').apply(lambda x: x[x['id6'] != 2]['id6'].sum())
print(analisis1.groupby('bucket')['id6'].agg([np.size]))


''' Ranguear en funciones de las calificaciones de riesgo definidas'''


#df2_base['ccal']=pd.cut(df2_base['puntaje'],bins=[-float('Inf'),835,941,1042,1104,1173
 #         ,float('Inf')], labels=['F','E','D','C','B','A'])


df2_base2['ccal']=pd.cut(df2_base2['puntaje'],bins=[-float('Inf'),835,941,1042,1104,1173
          ,float('Inf')], labels=['F','E','D','C','B','A'])

## Importante para analisis de toda la base (LGD EAD)  hay que bucvkear la df2_base con pd cut y los segmentos definidos por Saúl 

''' LGD EAD'''

analisis1=df2_base2[df2_base2['id0']!=1]
tab=analisis1.groupby('ccal')['saldo_gru','saldo_gru6'].sum()
tab['EADLGD']=tab['saldo_gru6']/tab['saldo_gru']

analisis1[analisis1.index.duplicated()]


''' Exportar a csv'''

exportar = df2_base[['ccodcta_gru','dfecrep','pd','puntaje',
                            'atraso0_cronograma_binned',
                                   'AVG_natrMax_binned',
                              #     'nAtrMax_gru_binned',
                                  'AVG_ccal_cli_binned',
                     'COMP_ATR_CLI_GRU_CONTABLE_binned',
                           'AVG_END_MAX_SOW_U3M_binned',
                    'AVG_EXP_NUM_MES_TOTAL_U12M_binned',
                    'ciclo_gru_binned'
       #'AVG_EXP_CANT_MES_PRIM_REF_MICRO_U12M_binned_NA',                
       ]]

exportar.to_csv('export2.csv', sep='|')

''' Revision de Variables'''

qq=train[['AVG_natrMax','id6']].compute()

qq.AVG_natrMax.hist()
qq['AVG_natrMax'].max()



''' Comprobación '''


x = 100
x=str(x)
f="select  ccodcta_gru ,dfecrep ,atraso0_cronograma, AVG_natrMax  ,nAtrMax_gru,AVG_ccal_cli,COMP_ATR_CLI_GRU_CONTABLE ,AVG_END_MAX_SOW_U3M,AVG_EXP_NUM_MES_TOTAL_U12M,ciclo_gru  from bdrie.dbo.mapeo1_vv "
#f="select top "+x+"  ccodcta_gru ,dfecrep ,atraso0_cronograma, AVG_natrMax  ,nAtrMax_gru,AVG_ccal_cli,COMP_ATR_CLI_GRU_CONTABLE ,AVG_END_MAX_SOW_U3M,AVG_EXP_NUM_MES_TOTAL_U12M,ciclo_gru  from bdrie.dbo.mapeo1_vv "
sql_con = pyodbc.connect('driver={SQL Server};SERVER=OF00SRVBDH;Trusted_Connection=True;DATABASE=MIS1')
query = f
comprobacion = pd.read_sql(query, sql_con)

comprobacion['COMP_ATR_CLI_GRU_CONTABLE'] = pd.to_numeric(comprobacion['COMP_ATR_CLI_GRU_CONTABLE']).copy()

comprobacion['atraso0_cronograma_binned']=w_atraso0_cronograma.deploy(comprobacion)  
comprobacion['AVG_natrMax_binned']=w_AVG_natrMax.deploy(comprobacion)  
comprobacion['nAtrMax_gru_binned']=w_nAtrMax_gru.deploy(comprobacion)  
comprobacion['AVG_ccal_cli_binned']=w_AVG_ccal_cli.deploy(comprobacion)  
comprobacion['COMP_ATR_CLI_GRU_CONTABLE_binned']=w_COMP_ATR_CLI_GRU_CONTABLE.deploy(comprobacion)  
comprobacion['AVG_END_MAX_SOW_U3M_binned']=w_AVG_END_MAX_SOW_U3M.deploy(comprobacion)  
comprobacion['AVG_EXP_NUM_MES_TOTAL_U12M_binned']=w_AVG_EXP_NUM_MES_TOTAL_U12M.deploy(comprobacion)  
comprobacion['ciclo_gru_binned']=w_ciclo_gru.deploy(comprobacion)  


comprobacion['puntaje']= (res.params[0]+ \
comprobacion['atraso0_cronograma_binned']*res.params[1]+ \
comprobacion['AVG_natrMax_binned']*res.params[2]+ \
#df2_base['nAtrMax_gru_binned']*res.params[3]+ \
comprobacion['AVG_ccal_cli_binned']*res.params[3]+ \
comprobacion['COMP_ATR_CLI_GRU_CONTABLE_binned']*res.params[4]+ \
comprobacion['AVG_END_MAX_SOW_U3M_binned']*res.params[5]+ \
comprobacion['AVG_EXP_NUM_MES_TOTAL_U12M_binned']*res.params[6]+ \
comprobacion['ciclo_gru_binned']*res.params[7] )  *80/math.log(2)+ 600
#df2_base['AVG_EXP_CANT_MES_PRIM_REF_MICRO_U12M_binned_NA']*res.params[8]) *80/math.log(2)+\


comprobacion2=comprobacion[[ 'atraso0_cronograma_binned', 'AVG_natrMax_binned',
       'AVG_ccal_cli_binned', 'COMP_ATR_CLI_GRU_CONTABLE_binned', 'AVG_END_MAX_SOW_U3M_binned',
       'AVG_EXP_NUM_MES_TOTAL_U12M_binned', 'ciclo_gru_binned']]     
   
comprobacion2 = sm.add_constant(comprobacion2)
comprobacion['pd']=res.predict(comprobacion2)          

comprobacion.groupby('dfecrep')['pd'].agg([np.mean])
  

''' Exportando a SQL'''

sql_con2 = pyodbc.connect('driver={SQL Server};SERVER=OF00SRVBDH;Trusted_Connection=True;DATABASE=BDRIE')            

cursor = sql_con2.cursor()

for index,row in df2_base.iterrows():
    #print(index,row)
    cursor.execute("INSERT INTO bdrie.dbo.testt([ccodcta_gru], \
    [dfecrep],[puntaje],[pd]) values (?, ?,?,?)", row['ccodcta_gru'],row['dfecrep'], row['puntaje'], row['pd']) 
    sql_con2.commit()
    
cursor.close()
sql_con2.close()


 

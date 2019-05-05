
# Importando librerias
import numpy as np
import pandas as pd

# Definiendo ruta de trabajo
#path='E:/CTIC/Big Data Analytics/NUEVO/'

# Importando base de datos
Xtrain=pd.read_csv('Xtrain.csv')
Xtest=pd.read_csv('Xtest.csv')
ytrain=pd.read_csv('ytrain.csv')
ytest=pd.read_csv('ytest.csv')

# Guardar el índice y eliminando columna
Xtrain.index=Xtrain['Unnamed: 0']
Xtrain.drop(['Unnamed: 0'],axis=1,inplace=True)

Xtest.index=Xtest['Unnamed: 0']
Xtest.drop(['Unnamed: 0'],axis=1,inplace=True)

ytrain.index=ytrain['Unnamed: 0']
ytrain.drop(['Unnamed: 0'],axis=1,inplace=True)

ytest.index=ytest['Unnamed: 0']
ytest.drop(['Unnamed: 0'],axis=1,inplace=True)

# Aplicar en test, las lógicas de train
#########################################

# 1) Valores extremos
#####################

# Asignando cotas inferiores
Xtest.loc[Xtest.edad<=21,'edad']=21
Xtest.loc[Xtest.deuda_sf<=0,'deuda_sf']=0

# Asignando cotas superiores
Xtest.loc[Xtest.edad>=63,'edad']=63
Xtest.loc[Xtest.deuda_sf>=57094.38,'deuda_sf']=57094.38

# 2) Missings
#############
np.sum(Xtest.isnull(),axis=0)
Xtest.exp_sf.fillna(28,inplace=True)
Xtest.linea_sf.fillna(0,inplace=True)
Xtest.deuda_sf.fillna(0,inplace=True)
np.sum(Xtest.isnull(),axis=0)

# 3) Tratamiento de variables categoricas
# Casa
Xtest['casa_f']=np.where(Xtest.casa=='ALQUILADA',0.5,
      np.where(Xtest.casa=='FAMILIAR',0.376174,
               np.where(Xtest.casa=='OTRAS',0.178571,0.558743)))
# Nivel educativo
base_dummy=pd.get_dummies(Xtest['nivel_educ'],prefix='d')
Xtest=pd.concat([Xtest,base_dummy],axis=1)
# Zona y Clasificación SBS
Xtest['zona_f']=np.where(Xtest.zona=='Lima',1,0)
Xtest['clasif_sbs_f']=np.where(Xtest.clasif_sbs==0,1,0)

# 4) Eliminando variables innecesarias
Xtest=Xtest.drop(['casa','nivel_educ','zona',
                  'clasif_sbs'],axis=1)

######################################################
######################################################
######################################################


log_edad=pd.DataFrame({'log_edad':np.log(Xtrain.edad)})
log_edad.hist()
# Regresión Logística
######################

# Importando librería
from sklearn.linear_model import LogisticRegression

# Generando objeto
logistic=LogisticRegression(random_state=4)
# Entrenamiento
logistic.fit(Xtrain,ytrain)
# Precisión en train
logistic.score(Xtrain,ytrain)
# Precisión en test
logistic.score(Xtest,ytest)
# Predicción
pr_logit=logistic.predict_proba(Xtrain)[:,1]
y_logit=logistic.predict(Xtrain)

# Tasa de aciertos
##################
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytrain,y_logit)
# Tasa de aciertos: 71.95%
print((cm[0,0]+cm[1,1])/np.sum(cm))
# Sensibilidad: 95.48%
print(cm[1,1]/np.sum(cm[1,:]))
# Especificidad: 16.31%
print(cm[0,0]/np.sum(cm[0,:]))

# Poder de discriminación: Gini
###############################
from sklearn.metrics import roc_auc_score
# Gini: 26.11%
2*roc_auc_score(ytrain,pr_logit)-1

# Calculando indicadores en test
################################
# Tasa de aciertos
cm=confusion_matrix(ytest,logistic.predict(Xtest))
# Tasa de aciertos: 71.2%
(cm[0,0]+cm[1,1])/np.sum(cm)
# Sensibilidad: 95.35%
cm[1,1]/np.sum(cm[1,:])
# Especificidad: 17.68%
cm[0,0]/np.sum(cm[0,:])
# Gini: 26.62%
2*roc_auc_score(ytest,logistic.predict_proba(Xtest)[:,1])-1

######################################################
######################################################
######################################################

# Naive Bayes
#############
from sklearn.naive_bayes import MultinomialNB # Libreria
Nbayes=MultinomialNB() # Objeto
Nbayes.fit(Xtrain,ytrain) # Entrenamiento
Nbayes.score(Xtrain,ytrain) # Precisión train
2*roc_auc_score(ytrain,Nbayes.predict_proba(Xtrain)[:,1])-1
Nbayes.score(Xtest,ytest) # Precisión test
2*roc_auc_score(ytest,Nbayes.predict_proba(Xtest)[:,1])-1

# Arbol de Clasificación
########################
from sklearn.tree import DecisionTreeClassifier
arbol_cl=DecisionTreeClassifier(max_depth=10) # Objeto
arbol_cl.fit(Xtrain,ytrain) # Entrenamiento
arbol_cl.score(Xtrain,ytrain) # Precisión train
2*roc_auc_score(ytrain,arbol_cl.predict_proba(Xtrain)[:,1])-1
arbol_cl.score(Xtest,ytest) # Precisión test
2*roc_auc_score(ytest,arbol_cl.predict_proba(Xtest)[:,1])-1

# Arbol de Regresión
####################
from sklearn.tree import DecisionTreeRegressor
arbol_reg=DecisionTreeRegressor(max_depth=10) # Objeto


# Random Forest Clasificación
#############################
from sklearn.ensemble import RandomForestClassifier
rf_cl=RandomForestClassifier(max_depth=10, n_estimators=10,
                             random_state=4)

# Random Forest Regresión
#############################
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(max_depth=10, n_estimators=10,
                             random_state=4)


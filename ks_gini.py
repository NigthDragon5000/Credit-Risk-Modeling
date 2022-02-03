
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np


#os.chdir("D:/")

#base  = pd.read_csv("b.csv")

def ks(test,pred):
    data =  pd.DataFrame({'bad':test, 'score':pred})
    data['good'] = 1 - data.bad
    rv1=data[data['bad']==1]['score']
    rv2=data[data['bad']==0]['score']
    return stats.ks_2samp(rv1, rv2)
     
#base.loc[base.PD_12M == 1, 'PD_12M'] = 3
#base.loc[base.PD_12M == 0, 'PD_12M'] = 1
#base.loc[base.PD_12M == 3, 'PD_12M'] = 0



def gini(test,pred,plot=False):
    
    
    logit_roc_auc_train = roc_auc_score(test, pred)
#fpr, tpr, thresholds = roc_curve(y_validation, logisticRegr.predict_proba(x_validation)[:,1])
    if plot:
        fpr, tpr, thresholds = roc_curve(test, pred)
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
        
    return logit_roc_auc_train,logit_roc_auc_train*2-1


def Find_Optimal_Cutoff(test, pred):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(test, pred)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 


def psi(df1,df2,var,nbreaks=10):
    '''
    df1: Muestra Original
    df2: Muestra a Probar
    var: Variable de interes
    nbreaks: Número de Cortes para el percentil
    Retorna una lista con [0] siendo la tabla y [1] el psi total'''
    #Cortamos en n percentiles
    breaks=pd.qcut(df1[var],nbreaks,duplicates='drop',retbins=True)[1]
    # Excluimos el mínimo y el máximo
    breaks=breaks[1:-1]
    # Rutina para añadir -Inf e Inf
    bins=[]
    bins.append(-float('Inf'))
    for i in breaks:
        bins.append(i)
    bins.append(float('Inf'))
    # Determinamos una etiqueta para cada fila
    q = pd.cut(df1[var], bins=bins,
                   labels=np.arange(len(bins)-1).astype(int))
    df1['labels']=q.astype(str)
    #Creamos tabla de cálculo
    col_names = {'amin':'minimo','amax':'maximo','size': 'obs'}
    tabla1 = df1.groupby(["labels"])[var].agg([np.min,np.max,np.size]).rename(columns=col_names).copy()
    tabla1['per_bin'] = tabla1['obs']/sum(tabla1['obs'])
    #Determinamos etiquetas para la segunda tabla
    q = pd.cut(df2[var], bins=bins,
                   labels=np.arange(len(bins)-1).astype(int))
    df2['labels']=q.astype(str)
    col_names = {'amin':'minimo','amax':'maximo','size': 'obs2'}
    tabla2 = df2.groupby(["labels"])[var].agg([np.min,np.max,np.size]).rename(columns=col_names).copy()
    tabla2['per_bin2'] = tabla2['obs2']/sum(tabla2['obs2'])
    #Uniendo en una sola tabla y calculando PSI
    tabla3=pd.concat([tabla1,tabla2[['obs2','per_bin2']]],axis=1)
    #a$PSI <- (a$relFreq-a$relFreq2)*log(a$relFreq/a$relFreq2)
    tabla3['psi']=(tabla3['per_bin']-tabla3['per_bin2'])*np.log(tabla3['per_bin']/tabla3['per_bin2'])
    return tabla3,sum(tabla3['psi'])


def massive_psi(df1,df2,nbreaks=10):
    psis=[]
    for column in df1.columns:
        try:
            res=psi(df1,df2,column,nbreaks)[1]
            psis.append([column,"{0:.6f}".format(res)])
        except KeyboardInterrupt:
            raise Exception('Stop by user')
        except: 
            pass
    return psis


''' Backward Elimination '''

def backwardElimination(Y, X, sl,frame=False,test=False,dftest=None):
    numVars = len(X.columns)
    for i in range(0, numVars):
        X = sm.add_constant(X)
        regressor = sm.Logit(Y, X).fit()
        if frame:
            print(regressor.summary())
        maxVar = max(regressor.pvalues)#.astype(float)
        if maxVar > sl:
            for name in regressor.pvalues.index:
                if (regressor.pvalues[name].astype(float) == maxVar) and name!='const': #\
               # and name!='const':
                    X=X.drop([name],axis=1)
                    if test:
                        dftest=dftest.drop([name],axis=1)
    return X,dftest


''' Funcion para eliminar correlacion '''

def eliminate_corr(df):
    corr=df.corr()
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if abs(corr.iloc[i,j]) == 1:
                if columns[j]:
                    columns[j] = False
    selected_columns = df.columns[columns]
    return selected_columns

''' Funcion para escribir xlsx'''

def save_xls(list_dfs, xls_path):
    with ExcelWriter(xls_path) as writer:
        for n, df in enumerate(list_dfs):
            df.to_excel(writer,'sheet%s' % n)
        writer.save()


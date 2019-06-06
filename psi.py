# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 09:18:19 2019

@author: jcondori
"""

import pandas as pd
import numpy as np

class psi():
    
    def __init__(self,bins=None,nbreaks=10):
        '''
        nbreaks: Número de Cortes para el percentil
        bins: Puntos de corte
        '''
        self.bins=bins
        self.nbreaks=nbreaks
        self.total_psi=None
    
    def fit(self,df1,df2,var):
        '''
        df1: Muestra Original
        df2: Muestra a Probar
        var: Variable de interes

        Retorna la tabla '''
        
        #Cortamos en n percentiles       
        if self.bins is None:
            breaks=pd.qcut(df1[var],self.nbreaks,duplicates='drop',retbins=True)[1]
            # Excluimos el mínimo y el máximo
            breaks=breaks[1:-1]
            # Rutina para añadir -Inf e Inf
            bins=[]
            bins.append(-float('Inf'))
            for i in breaks:
                bins.append(i)
            bins.append(float('Inf'))
        else:
            bins=self.bins
        
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
        tabla3['psi']=(tabla3['per_bin']-tabla3['per_bin2'])*np.log(tabla3['per_bin']/tabla3['per_bin2'])
        
        #Guardamos en la propiedad del objeto
        self.total_psi=sum(tabla3['psi'])
        
        return tabla3


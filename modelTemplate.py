import os
import numpy as np
import pandas as pd
os.chdir('C:\\Users\\pc\\Documents\\PythonML')
os.chdir('C:\\Users\\pc\\Documents')

dataset = pd.read_csv('Social_Network_Ads.csv')
dataset.iloc[0,2]= None
dataset.iloc[7,2]= None

from woe3 import woe

w=woe(nbreaks=4)
w.massive(dataset,'Purchased')

w=woe(nbreaks=4)
w.fit(dataset['Age'],dataset['Purchased'])
w.optimize()  
w.plot()      
dataset['Age_binning']=w.deploy(dataset)    

ww=woe(nbreaks=4)
ww.fit(dataset['EstimatedSalary'],dataset['Purchased'])
ww.optimize(depth=1)  
ww.plot()
dataset['Salary_binning']=ww.deploy(dataset)  

''' Data Exploration'''

dataset['Purchased'].value_counts()

#Proportion of Good Bad
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='Purchased',data=dataset,palette='hls')
plt.show()

# Histogram of Age
dataset.Age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('hist_age')

sns.distplot(pru['Age'], hist=True, kde=True, 
               color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

# Two Histogramns
pru=dataset.dropna()
pru1=pru[pru['Purchased']==1]
pru2=pru[pru['Purchased']==0]
bins = np.linspace(15, 60, 20)
plt.hist(pru1['Age'].values,bins, alpha=0.3, label='Compra')
plt.hist(pru2['Age'].values,bins, alpha=0.3, label='NoCompra')
plt.legend(loc='upper right')
plt.show()



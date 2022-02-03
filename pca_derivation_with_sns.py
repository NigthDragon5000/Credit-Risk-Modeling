
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

iris = sns.load_dataset("iris")

df = iris

df=df.dropna()

X=df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

Y =df[['species']]

#x_std=X

x_std=StandardScaler().fit_transform(X)

features=x_std.T

cov = np.cov(features)

print(cov)

eig_vals,eig_vecs=np.linalg.eig(cov)


eig_vals[0]/sum(eig_vals)

projected_X = x_std.dot(eig_vecs[:,0:2])

result=pd.DataFrame(projected_X)
result.columns=['PCA1','PCA2']
result['y_axis']=0
result['label']=Y
#result[result.label.isna()==True]=0

sns.lmplot('PCA1','PCA2',data=result,fit_reg=False,
           scatter_kws={"s":50},
           hue="label")
plt.title('PCA Result')

'''
'''

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid")
iris = sns.load_dataset("iris")


# Subset the iris dataset by species
setosa = iris.query("species == 'setosa'")
virginica = iris.query("species == 'virginica'")

# Set up the figure
f, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect("equal")

# Draw the two density plots
ax = sns.kdeplot(setosa.sepal_width, setosa.sepal_length,
                 cmap="Reds", shade=True, shade_lowest=False)
ax = sns.kdeplot(virginica.sepal_width, virginica.sepal_length,
                 cmap="Blues", shade=True, shade_lowest=False)

# Add labels to the plot
red = sns.color_palette("Reds")[-2]
blue = sns.color_palette("Blues")[-2]
ax.text(2.5, 8.2, "virginica", size=16, color=blue)
ax.text(3.8, 4.5, "setosa", size=16, color=red)


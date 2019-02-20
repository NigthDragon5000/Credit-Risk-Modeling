from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from ks_gini import ks,gini,Find_Optimal_Cutoff

mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=1000, 
                    activation='logistic')

mlp.fit(x_train,y_train)

prob=mlp.predict_proba(x_test)[:,1]

ks(y_test,prob)
gini(y_test,prob,plot=True)



''' Defining Cutoff'''
Find_Optimal_Cutoff(y_test,prob)

predictions = (mlp.predict_proba(x_test)[:,1] >= 0.9748).astype(int)

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))

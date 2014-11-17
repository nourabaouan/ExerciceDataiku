import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import cross_validation

Fichier = pd.read_csv("census_income_learn.csv")
i=0
N=2
tauxDeReconnaissance=np.zeros((1,4), dtype='i')
Risques=np.zeros((1,4), dtype='i')
Size=Fichier.shape
nbEchantillons=(Size[0])/N
#for i in range(0,N):
Dtest=Fichier.ix[i*nbEchantillons:(i+1)*nbEchantillons,:]
Dtesta=Dtest
Dlearn=Fichier.drop(Fichier.index[i*nbEchantillons:(i+1)*nbEchantillons], axis=0)
# encode labels
labels = Dlearn.ix[:,41] 
le = preprocessing.LabelEncoder()
labels_fea = le.fit_transform(labels) 
# vectorize training data
Dlearn=Dlearn.drop(Dlearn.columns[41], axis=1)
Dlearn_as_dicts = [dict(r.iteritems()) for _, r in Dlearn.iterrows()]
Dlearn_fea = DictVectorizer(sparse=False).fit_transform(Dlearn_as_dicts)
# use decision tree
dt = tree.DecisionTreeClassifier()
arbre=dt.fit(Dlearn_fea, labels_fea)	
#partie Test
Dtest=Dtest.drop(Dtest.columns[41], axis=1)
Dtest_as_dictsTest = [dict(r.iteritems()) for _, r in Dtest.iterrows()]
Dtest_feaTest = DictVectorizer(sparse=False).fit_transform(Dtest_as_dictsTest)
###Test
predictions=arbre.predict(Dtest_feaTest)
predictionsS = le.inverse_transform(predictions.astype('I'))
predictions_as_dataframe = Dtest.join(pd.DataFrame({"Prediction": predictionsS}))	
######Taux de reconnaissance ava
s=predictions_as_dataframe.shape
maxi=max(s)
print(maxi)
l=0
for j in range(1,maxi):
	if(predictions_as_dataframe.ix[j,41]==Dtesta.ix[j,41]): l+=1
l=float(l)
maxi=float(maxi)
tauxDeReconnaissance[0][i]=l/maxi
Risques[0][i]=1-tauxDeReconnaissance[0][i]
print(l/maxi)	
	
	

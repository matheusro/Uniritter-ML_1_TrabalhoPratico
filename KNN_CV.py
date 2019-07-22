import pandas as pd
from sklearn.model_selection import cross_val_score 
from sklearn.neighbors import KNeighborsClassifier 
 
dataset = pd.read_csv('C:/dev/Uniritter_DS/MachineLearning_I/Uniritter-ML_1_TrabalhoPratico/data/dataset.csv', delimiter=',',  low_memory=False)
X = dataset.drop(['Cover_Type'], axis=1)
y = dataset['Cover_Type']


KNN_classifier = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
          metric_params=None, n_jobs=None, n_neighbors=3, p=2,
          weights='distance')  
''' PARAMETROS PARA TUNAR
  algorithm='auto', leaf_size=30, metric='euclidean',
          metric_params=None, n_jobs=None, n_neighbors=8, p=2,
          weights='uniform'
'''
KNN_classifier.fit(X, y) 
knn_cv_score_AC = cross_val_score(KNN_classifier, X, y, cv=10, scoring='accuracy')
print("Acuracia KNN Com CV:", knn_cv_score_AC.mean())  



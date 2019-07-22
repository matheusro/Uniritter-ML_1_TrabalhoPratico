import pandas as pd
#from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier 
 
dataset = pd.read_csv('C:/dev/Uniritter_DS/MachineLearning_I/Uniritter-ML_1_TrabalhoPratico/data/dataset.csv', delimiter=',',  low_memory=False)

X = dataset.drop(['Cover_Type'], axis=1)
y = dataset['Cover_Type']


KNN_classifier = KNeighborsClassifier()  
KNN_classifier.fit(X, y) 

#knn_cv_score_AC = cross_val_score(KNN_classifier, X, y, cv=10, scoring='accuracy')


#print("Acuracia KNN Com CV:", knn_cv_score_AC.mean())  

grid_params = {'n_neighbors': [3,5,11,19],
               'weights': ['uniform','distance'],
               'metric': ['euclidean', 'manhattan']
               }
gs = GridSearchCV(KNeighborsClassifier(),
                  
                  grid_params,
                  verbose=1,
                  cv=10,
                  n_jobs=-1
                  )

gs_results = gs.fit(X,y)
#gs_results_estimator = gs_results.best_estimator_
gs_results_params = gs_results.best_params_


#print("Best KNN Estimator: {}".format(gs_results_estimator))
print("Tuned Best KNN Parameters: {}".format(gs_results_params))
print("Best KNN Score is {}".format(gs_results.best_score_))
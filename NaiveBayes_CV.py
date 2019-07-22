import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB


dataset = pd.read_csv('C:/dev/Uniritter_DS/MachineLearning_I/Uniritter-ML_1_TrabalhoPratico/data/dataset.csv', delimiter=',',  low_memory=False)
X = dataset.drop(['Cover_Type'], axis=1)
y = dataset['Cover_Type']

naive = GaussianNB()

naive.fit(X, y)

naive_cv_score_AC = cross_val_score(naive, X, y, cv=10, scoring='accuracy')

print("Acuracia Naive Com CV:", naive_cv_score_AC.mean())  
 




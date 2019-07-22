import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import cross_val_score 

 
dataset = pd.read_csv('C:/dev/Uniritter_DS/MachineLearning_I/Uniritter-ML_1_TrabalhoPratico/data/dataset.csv', delimiter=',',  low_memory=False)

X = dataset.drop(['Cover_Type'], axis=1)
y = dataset['Cover_Type']

Tree_classifier = DecisionTreeClassifier()  
Tree_classifier.fit(X, y) 

scores = cross_val_score(Tree_classifier, X, y, cv=10, scoring='accuracy')


print("Acuracia Decision Com CV:",scores.mean()) 


import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import GridSearchCV

 
dataset = pd.read_csv('C:/dev/Uniritter_DS/MachineLearning_I/Uniritter-ML_1_TrabalhoPratico/data/dataset.csv', delimiter=',',  low_memory=False)

X = dataset.drop(['Cover_Type'], axis=1)
y = dataset['Cover_Type']

# Creating the hyperparameter grid 
grid_params_tree = {"max_depth": [3, 5, 10],
              "max_features": [ 3, 5, 10, 15, 20],
              "min_samples_leaf": [1, 3],
              "criterion": ["gini", "entropy"]
              }
 

# Instantiating Decision Tree classifier
tree = DecisionTreeClassifier()  

# Instantiating RandomizedSearchCV object
tree_cv = GridSearchCV(tree, grid_params_tree, cv = 10)  

tree_cv.fit(X, y)  

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))

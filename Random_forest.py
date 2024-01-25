import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def Random_forest(X_train,y_train,X_test,pd,y_test):

    rfc = RandomForestClassifier( n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)

    y_pred = rfc.predict(X_test)

    print(classification_report(y_test, y_pred))



    param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    }

    # lot of time to run

    #rf = RandomForestClassifier()

   # grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')

    #grid_search.fit(X_train, y_train)
    #best_params = grid_search.best_params_

    #best_model = grid_search.best_estimator_

    #predictions = best_model.predict(X_test)

    #accuracy = accuracy_score(y_test, predictions)
    #report = classification_report(y_test, predictions)

 #   print(f"Meilleurs paramètres : {best_params}")
  #  print(f'Accuracy avec meilleurs paramètres: {accuracy}')
   # print('Classification Report avec meilleurs paramètres:\n', report)
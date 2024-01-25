from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def XGB_algorithm(X_train,y_train,X_test,pd,y_test):

    model = XGBClassifier(objective='multi:softmax', num_class=3, seed=42)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f'Accuracy: {accuracy}')
    print('Classification Report:\n', report)


    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }

    xgb = XGBClassifier(objective='multi:softmax', num_class=3, seed=42)
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f"Meilleurs paramètres : {best_params}")
    print(f'Accuracy avec meilleurs paramètres: {accuracy}')
    print('Classification Report avec meilleurs paramètres:\n', report)
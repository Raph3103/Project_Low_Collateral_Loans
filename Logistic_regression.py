from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def Logistic_regression(X_train,y_train,X_test,pd,y_test):

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print(f'Accuracy: {accuracy}')
    print('Classification Report:\n', report)



    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
    }

    logistic_regression = LogisticRegression(max_iter=1000)

    grid_search = GridSearchCV(logistic_regression, param_grid, cv=5, scoring='accuracy')

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_


    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print(f"Meilleurs paramètres : {best_params}")
    print(f'Accuracy avec meilleurs paramètres: {accuracy}')
    print('Classification Report avec meilleurs paramètres:\n', report)
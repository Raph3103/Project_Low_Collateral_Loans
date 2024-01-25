
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def Knn_algorithm(X_train,y_train,X_test,pd,y_test):

    knn = KNeighborsClassifier(n_neighbors=3)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    print('Printing informations')
    print( 'ypred:', (y_pred))
    #print( 'y_train:', y_train)
    #print( 'X_test:', X_test)
    # Convertir les prédictions en DataFrame pandas
    predictions_df = pd.DataFrame(y_pred, columns=['Prediction'])
    information_df = pd.DataFrame(X_test,columns=['X_test'])
    combined_df = pd.concat([information_df, predictions_df], axis=1)
    # Exporter le DataFrame en fichier CSV
    combined_df.to_csv('mes_predictions.csv', index=False)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)
    df__ = pd.DataFrame(X_test)
    df__.to_csv('mon_fichier.csv', index=False)


    # testing for best performance



    param_grid = {
        'n_neighbors': [3],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2],
    }
    knn = KNeighborsClassifier()

    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f"Meilleurs paramètres : {best_params}")
    print(f'Accuracy avec meilleurs paramètres: {accuracy}')
    print('Classification Report avec meilleurs paramètres:\n', report)


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def Logistic_model(X_train , X_test, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    Logistic_model_pred = model.predict(X_test)
    return model,Logistic_model_pred

def KNN_model(X_train, X_test, y_train ):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    KNN_model_pred = model.predict(X_test)
    return model, KNN_model_pred
from sklearn.metrics import accuracy_score

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    return accuracy_score(y, y_pred)
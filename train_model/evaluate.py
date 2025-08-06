from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score,classification_report

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    return {
        'accuracy':accuracy_score(y, y_pred)*100,
        'precision':precision_score(y, y_pred)*100,
        'recall':recall_score(y, y_pred)*100,
        'f1_score':f1_score(y, y_pred)*100,
        'y_true':y,
        'y_pred':y_pred
    }
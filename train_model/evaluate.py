import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

def evaluate_model(model, X_test, y_test):
    # Predict and coerce to {0,1} ints
    y_pred = model.predict(X_test)
    y_true = np.asarray(y_test).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)

    # Confusion matrix in fixed order [0,1] = [Healthy, Tumor]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Binary (tumor=1) metrics via sklearn — identical to TP/FP/TN/FN formulas
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    rec  = recall_score(y_true, y_pred,  average="binary", pos_label=1, zero_division=0)
    f1   = f1_score(y_true, y_pred,     average="binary", pos_label=1, zero_division=0)

    total_obs = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    n_neg = total_obs - n_pos

    return {
        'accuracy': acc * 100,
        'precision': prec * 100,
        'recall': rec * 100,
        'f1_score': f1 * 100,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'n_observations': int(total_obs),
        'n_label_positives': int(n_pos),
        'n_label_negatives': int(n_neg),
        'y_true': y_true,
        'y_pred': y_pred,
        'confusion_matrix': cm.tolist()
    }


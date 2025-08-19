import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

def evaluate_model(model, X_test, y_test):
    # --- Predict and coerce shapes/types ---
    y_pred = model.predict(X_test)
    y_true = np.asarray(y_test).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)

    # --- Full report with explicit label mapping to names ---
    report = classification_report(
        y_true, y_pred,
        labels=[0, 1],                        # enforce 0->Healthy, 1->Tumor
        target_names=['Healthy','Tumor'],
        output_dict=True,
        zero_division=0
    )

    # Demonstration only (NOT equal to f1_w in general):
    acc      = accuracy_score(y_test, y_pred) * 100
    p_w      = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    r_w      = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    # Manual harmonic mean (like "normal" F1 you expect)
    f1_harmonic = (2 * (p_w/100) * (r_w/100) / ((p_w/100)+(r_w/100))) * 100 if (p_w + r_w) > 0 else 0.0


    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total_obs = len(y_test)
    pos_labels = int(sum(y_test))      # actual positives
    neg_labels = total_obs - pos_labels

    return {
        'accuracy': acc,
        'precision': p_w,
        'recall': r_w,
        'f1_score': f1_harmonic,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'n_observations': total_obs,
        'n_label_positives': pos_labels,
        'n_label_negatives': neg_labels,
        'y_true': y_test,
        'y_pred': y_pred,
        'confusion_matrix': cm.tolist()
    }

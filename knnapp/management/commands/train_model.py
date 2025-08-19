from django.core.management.base import BaseCommand
import os, re, collections
import joblib
import numpy as np
from train_model.data_loader import load_data, extract_subject_id
from train_model.knn_trainer import train_knn
from sklearn.model_selection import GroupShuffleSplit
from train_model.evaluate import evaluate_model
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from django.conf import settings
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from typing import Tuple

class Command(BaseCommand):
    help = 'Trains the KNN model on MRI brain image data and saves it to disk.'

    def handle(self, *args, **kwargs):
        healthy_dir = "/Volumes/TOSHIBA EXT/Healthy_Brain_Images"
        tumor_dir = "/Volumes/TOSHIBA EXT/Tumor_Brain_Images"

        self.stdout.write("📦 Loading data...")
        X, y, paths = load_data(healthy_dir, tumor_dir)

        # ---------- Audit labels vs path ----------
        y_check = np.array([label_from_path(p, healthy_dir, tumor_dir) for p in paths])
        mismatch_idx = np.where(y_check != y)[0]
        print("Label mismatches (path vs y):", len(mismatch_idx))
        if len(mismatch_idx) > 0:
            print("FIRST mismatches:", [(paths[i], int(y[i]), int(y_check[i])) for i in mismatch_idx[:10]])
            # Enforce path-based labels (robust)
            y = y_check

        # Build groups EXACTLY aligned to X rows
        groups = np.array([extract_subject_id(p) for p in paths])
        # --- DIAGNOSTIC: subject counts per class before any collapse ---
        uniq_subj = np.unique(groups)
        # per-subject label = label of first occurrence of that subject
        subj_to_label = {}
        for s in uniq_subj:
            i = np.where(groups == s)[0][0]
            subj_to_label[s] = int(y[i])

        counts_by_class = {}
        for cls in np.unique(y):
            counts_by_class[int(cls)] = sum(1 for s in uniq_subj if subj_to_label[s] == cls)

        print("Subjects per class BEFORE collapse:", counts_by_class)
        # Optional: sample a few extracted IDs and stems to check parser quality
        print("Sample (subject_id, path, label):",
        [(extract_subject_id(paths[i]), os.path.basename(paths[i])[:60], int(y[i])) for i in range(min(10, len(paths)))])
        # ---------- Optional collapse: one file per subject (reduces near-duplicates) ----------
        # Decide whether collapsing is safe (need >= 2 subjects per class)
        class_ids = np.unique(y)
        subjects_per_class = {int(c): len(set(groups[y == c])) for c in class_ids}
        print("Subjects per class (for collapse decision):", subjects_per_class)

        use_collapse = all(subjects_per_class.get(int(c), 0) >= 2 for c in class_ids)

        if use_collapse:
            try:
                paths_subj1, y_subj1, groups_subj1, keep_idx = one_file_per_subject(
                    paths, y, groups, per_class_max_subjects=50, rng_seed=42
                )
                X_subj1 = X[keep_idx]
                print("Collapse succeeded:",
                      {int(c): len(set(groups_subj1[y_subj1 == c])) for c in np.unique(y_subj1)})
            except ValueError as e:
                print("Collapse skipped due to:", e)
                paths_subj1, y_subj1, groups_subj1, X_subj1 = paths, y, groups, X
        else:
            print("Not enough subjects to collapse safely; using all files without collapse.")
            paths_subj1, y_subj1, groups_subj1, X_subj1 = paths, y, groups, X

        # ---------- Grouped + stratified split (robust) ----------
        train_idx, test_idx = grouped_stratified_split_robust(
            X_subj1, y_subj1, groups_subj1, test_size=0.2, random_state=42
        )
        X_train, X_test = X_subj1[train_idx], X_subj1[test_idx]
        y_train, y_test = y_subj1[train_idx], y_subj1[test_idx]

        # Quick checks
        train_subj = set(groups_subj1[train_idx]); test_subj = set(groups_subj1[test_idx])
        print("Subject overlap count:", len(train_subj & test_subj))
        print("Train classes:", np.unique(y_train), "Test classes:", np.unique(y_test))
        

        self.stdout.write("🤖 Training model...")
        model = train_knn(X_train, y_train)

        self.stdout.write("📊 Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        plot_and_save_graphs(metrics)
        
        #Remove y_true and y_pred before saving
        metrics_to_save = {
            k:v for k, v in metrics.items() if k not in ('y_true', 'y_pred')
        }
        model_path = os.path.join("train_model","trained_knn_model.joblib")
        joblib.dump((model, metrics_to_save), model_path)

        print(y_train)
        print(y_test)
        print(X_test)

        self.stdout.write(self.style.SUCCESS(
            (
                "✅ Model trained and saved successfully.\n"
                f"Accuracy: {metrics['accuracy']:.2f}%\n"
                f"Recall: {metrics['recall']:.2f}%\n"
                f"Precision: {metrics['precision']:.2f}%\n"
                f"F1 Score (harmonic mean): {metrics['f1_score']:.2f}%\n\n"
                f"True Positives: {metrics['true_positives']}\n"
                f"False Positives: {metrics['false_positives']}\n"
                f"True Negatives: {metrics['true_negatives']}\n"
                f"False Negatives: {metrics['false_negatives']}\n"
                f"Total Observations: {metrics['n_observations']}\n"
                f"Labeled Positives: {metrics['n_label_positives']}\n"
                f"Labeled Negatives: {metrics['n_label_negatives']}"
            )
        ))

def one_file_per_subject(paths, y, groups, per_class_max_subjects=50, rng_seed=42):
    """
    Select at most one file per subject and cap #subjects per class.
    Returns (paths_f, y_f, groups_f, keep_indices) aligned to the *original* arrays.
    """
    import random
    rng = random.Random(rng_seed)

    # subject -> indices in the original arrays
    by_subj = {}
    for i, g in enumerate(groups):
        by_subj.setdefault(g, []).append(i)

    # For each subject, choose a subject-level label via majority vote (ties -> first)
    subj_label = {}
    for g, idxs in by_subj.items():
        labels = [int(y[i]) for i in idxs]
        if len(set(labels)) > 1:
            lab = 1 if sum(labels) >= (len(labels) / 2) else 0
        else:
            lab = labels[0]
        subj_label[g] = lab

    # Subjects per class
    subj_0 = [s for s in by_subj if subj_label[s] == 0]
    subj_1 = [s for s in by_subj if subj_label[s] == 1]
    rng.shuffle(subj_0)
    rng.shuffle(subj_1)
    subj_0 = subj_0[:per_class_max_subjects]
    subj_1 = subj_1[:per_class_max_subjects]

    if len(subj_0) < 2 or len(subj_1) < 2:
        raise ValueError(
            f"Not enough subjects per class after collapsing. healthy={len(subj_0)}, tumor={len(subj_1)}. "
            f"Add more subjects or reduce per_class_max_subjects."
        )

    # Keep one representative file per subject (first index)
    keep_indices = []
    for s in subj_0 + subj_1:
        keep_indices.append(by_subj[s][0])

    keep_indices = sorted(keep_indices)
    paths_f  = [paths[i] for i in keep_indices]
    y_f      = np.array([y[i] for i in keep_indices])
    groups_f = np.array([groups[i] for i in keep_indices])
    return paths_f, y_f, groups_f, np.array(keep_indices)

def plot_and_save_graphs(metrics: dict) -> None:
    """
    Save bar chart of weighted metrics and a confusion matrix heatmap
    to knnapp/static/images/*.png using absolute paths.
    Expects metrics to contain:
      - 'accuracy' (percentage, 0..100)
      - 'precision_weighted', 'recall_weighted', 'f1_weighted' (percentages)
      - 'y_true', 'y_pred' (1D arrays with labels 0=Healthy, 1=Tumor)
    """
    # 1) Ensure output directory exists (ABSOLUTE path)
    output_dir = os.path.join(settings.BASE_DIR, "knnapp", "static", "images")
    os.makedirs(output_dir, exist_ok=True)

    metrics_path = os.path.join(output_dir, "eval_metrics.png")
    cm_path      = os.path.join(output_dir, "confusion_matrix.png")

    # 2) Build metric series using existing keys
    #    (All are already percentages per your evaluate_model.)
    names  = ["Accuracy", "Precision (w)", "Recall (w)", "F1 (w)"]
    scores = [
        float(metrics.get("accuracy", 0.0)),
        float(metrics.get("precision_weighted", 0.0)),
        float(metrics.get("recall_weighted", 0.0)),
        float(metrics.get("f1_weighted", 0.0)),
    ]

    # 3) Plot bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(names, scores)
    plt.title("Model Evaluation Metrics (Weighted)")
    plt.ylabel("Percentage")
    plt.ylim(0, 100)  # because values are already *percentages*
    # Optional: annotate bars with values
    for i, v in enumerate(scores):
        plt.text(i, min(v + 2, 100), f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(metrics_path, dpi=150)
    plt.close()

    # 4) Confusion matrix
    y_true = metrics.get("y_true", [])
    y_pred = metrics.get("y_pred", [])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Choose tick labels by convention
    tick_labels = ["Healthy", "Tumor"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=tick_labels, yticklabels=tick_labels
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

def grouped_stratified_split_robust(X: np.ndarray,
                                    y: np.ndarray,
                                    groups: np.ndarray,
                                    test_size: float = 0.2,
                                    random_state: int = 42):
    """
    Prefer stratifying at the subject level if each class has >=2 subjects.
    Otherwise, fall back to a grouped split that ensures at least one subject
    of each class is in test. No subject leakage either way.
    """
    # Unique subjects and their per-subject label (take label of first sample)
    unique_subj, subj_inv = np.unique(groups, return_inverse=True)
    subj_labels = np.zeros(len(unique_subj), dtype=y.dtype)
    for i in range(len(unique_subj)):
        first_idx = np.where(subj_inv == i)[0][0]
        subj_labels[i] = y[first_idx]

    classes, counts = np.unique(subj_labels, return_counts=True)

    def indices_from_subjects(subj_subset):
        mask = np.isin(groups, subj_subset)
        return np.where(mask)[0]

    # Case A: feasible stratification
    if np.all(counts >= 2):
        subj_train, subj_test = train_test_split(
            unique_subj,
            test_size=test_size,
            random_state=random_state,
            stratify=subj_labels
        )
        train_idx = indices_from_subjects(subj_train)
        test_idx  = indices_from_subjects(subj_test)
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), "Subject leakage!"
        assert len(np.unique(y[test_idx])) > 0
        return train_idx, test_idx

    # Case B: fallback grouped split that guarantees both classes in test (if both exist)
    rng = np.random.RandomState(random_state)
    by_class = {c: unique_subj[subj_labels == c] for c in classes}

    desired_test_n = max(1, int(round(test_size * len(unique_subj))))
    forced_test = []

    for c, arr in by_class.items():
        if len(arr) >= 1:
            chosen = rng.choice(arr, size=1, replace=False)
            forced_test.extend(chosen.tolist())

    forced_test = np.array(sorted(set(forced_test)))
    remaining_subjects = np.setdiff1d(unique_subj, forced_test, assume_unique=False)

    n_to_add = max(0, desired_test_n - len(forced_test))
    if n_to_add > 0 and len(remaining_subjects) > 0:
        add = rng.choice(remaining_subjects, size=min(n_to_add, len(remaining_subjects)), replace=False)
        subj_test = np.array(sorted(set(np.concatenate([forced_test, add]))))
    else:
        subj_test = forced_test

    subj_train = np.setdiff1d(unique_subj, subj_test, assume_unique=False)

    train_idx = indices_from_subjects(subj_train)
    test_idx  = indices_from_subjects(subj_test)

    # Safety
    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), "Subject leakage!"

    # Try to ensure both classes in test if both exist overall
    if len(np.unique(y[test_idx])) < 2 and len(classes) >= 2:
        missing = [c for c in classes if c not in np.unique(y[test_idx])]
        if missing:
            missing_class = missing[0]
            # find a subject of the missing class in train and move it
            # NOTE: compute subj_labels map
            subj_label_map = {s: subj_labels[np.where(unique_subj == s)[0][0]] for s in unique_subj}
            candidates = [s for s in subj_train if subj_label_map[s] == missing_class]
            if len(candidates) > 0:
                move = rng.choice(candidates, size=1, replace=False)[0]
                subj_train = np.setdiff1d(subj_train, [move])
                subj_test  = np.array(sorted(set(np.concatenate([subj_test, [move]]))))
                train_idx = indices_from_subjects(subj_train)
                test_idx  = indices_from_subjects(subj_test)
                assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))

    return train_idx, test_idx

def label_from_path(p: str, healthy_root: str, tumor_root: str) -> int:
    p_norm = os.path.abspath(p)
    h_root = os.path.abspath(healthy_root)
    t_root = os.path.abspath(tumor_root)
    # Use commonpath to test ancestry robustly
    if os.path.commonpath([p_norm, h_root]) == h_root:
        return 0
    if os.path.commonpath([p_norm, t_root]) == t_root:
        return 1
    raise ValueError(f"Path {p} is not under healthy or tumor roots")

# knn_app/knn_model/train.py

import os
import nibabel as nib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

def extract_features_from_nii(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    flattened = data.flatten()
    return flattened[:10000]  # Reduce dimensionality

def train_knn_model(healthy_dir, tumor_dir, save_path='knn_model.pkl'):
    X, y = [], []

    for file in os.listdir(healthy_dir):
        if file.endswith(".nii.gz"):
            X.append(extract_features_from_nii(os.path.join(healthy_dir, file)))
            y.append(0)  # 0 = healthy

    for file in os.listdir(tumor_dir):
        if file.endswith(".nii.gz"):
            X.append(extract_features_from_nii(os.path.join(tumor_dir, file)))
            y.append(1)  # 1 = tumor

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    joblib.dump(model, save_path)
    print(f"KNN model saved to {save_path}")

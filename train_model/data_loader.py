import os
import numpy as np
import nibabel as nib
from sklearn.utils import shuffle

def load_data(healthy_dir, tumor_dir):
    X = []
    y = []

    for root, _, files in os.walk(healthy_dir):
        for file in files:
            if file.endswith(".nii.gz"):
                path = os.path.join(root, file)
                img = nib.load(path).get_fdata()
                X.append(img.flatten())
                y.append(0)  # healthy

    for root, _, files in os.walk(tumor_dir):
        for file in files:
            if file.endswith(".nii.gz"):
                path = os.path.join(root, file)
                img = nib.load(path).get_fdata()
                X.append(img.flatten())
                y.append(1)  # tumor

    X, y = shuffle(X, y, random_state=42)
    return np.array(X), np.array(y)
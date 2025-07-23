# knnapp/ml/knn_model.py

import numpy as np
import nibabel as nib
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path

def load_image_as_array(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return np.mean(data, axis=(0, 1, 2)).reshape(1, -1)  # Flattened 1D feature

def train_knn_model(images_queryset):
    features, labels = [], []
    for obj in images_queryset:
        try:
            feature = load_image_as_array(obj.image_file.path)
            features.append(feature.flatten())
            labels.append(obj.label)
        except Exception as e:
            print(f"Error processing {obj.image_file.path}: {e}")
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(features, labels)
    return "Model trained on {} samples.".format(len(labels))

def classify_image(new_image_path, model):
    feature = load_image_as_array(new_image_path)
    prediction = model.predict(feature)
    return prediction[0]

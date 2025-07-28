from django.shortcuts import render
from train_model.data_loader import load_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import joblib

def index(request):
    model_path = os.path.join("train_model", "knn_model.joblib")
    metrics = {}

    if request.method == 'POST' and 'train_model' in request.POST:
        healthy_dir = '/Volumes/TOSHIBA EXT/Healthy_Brain_Images'
        tumor_dir = '/Volumes/TOSHIBA EXT/Tumor_Brain_Images'

        # Load data
        X, y = load_data(healthy_dir, tumor_dir, max_per_class=50)

        # Split: 80% train, 20% test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)

        # Evaluate
        y_pred = knn.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Save metrics
        metrics = {
            'accuracy': round(report['accuracy'] * 100, 2),
            'precision': round(report['weighted avg']['precision'] * 100, 2),
            'recall': round(report['weighted avg']['recall'] * 100, 2),
            'f1_score': round(report['weighted avg']['f1-score'] * 100, 2)
        }

        # Save model and metrics
        joblib.dump((knn, metrics), model_path)

    elif os.path.exists(model_path):
        _, metrics = joblib.load(model_path)

    return render(request, 'knn_app/index.html', {'metrics': metrics})



# train.py
from django.core.management.base import BaseCommand
from train_model.data_loader import load_data
from sklearn.neighbors import KNeighborsClassifier
from django.shortcuts import render
import joblib
import os


def index(request):
    model_path = os.path.join("knnapp", "train_model", "knn_model.joblib")

    if os.path.exists(model_path):
        knn, accuracy = joblib.load(model_path)
        context = {
            'model_trained': True,
            'accuracy': f"{accuracy * 100:.2f}%"
        }
    else:
        context = {
            'model_trained': False,
            'accuracy': None
        }

    return render(request, 'knnapp/index.html', context)


class Command(BaseCommand):
    help = 'Train KNN model on MRI data'

    def handle(self, *args, **kwargs):
        print("📦 Loading data...")
        
        healthy_dir = '/Volumes/ExternalDrive/Healthy_Brain_Images'
        tumor_dir = '/Volumes/ExternalDrive/Tumor_Brain_Images'

        # Load the sample data
        X, y = load_data(healthy_dir, tumor_dir, max_per_class=50)

        print("✅ Data loaded. Training model...")

        # Train KNN model
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X, y)

        # Evaluate accuracy on training data (optional for now)
        accuracy = knn.score(X, y)

        # Save model and accuracy
        model_path = os.path.join("knnapp", "train_model", "knn_model.joblib")
        joblib.dump((knn, accuracy), model_path)

        print(f"✅ Model trained and saved to {model_path} with accuracy: {accuracy * 100:.2f}%")



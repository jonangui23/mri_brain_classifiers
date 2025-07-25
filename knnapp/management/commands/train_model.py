from django.core.management.base import BaseCommand
import os
import joblib
from train_model.data_loader import load_data
from train_model.knn_trainer import train_knn
from train_model.evaluate import evaluate_model

class Command(BaseCommand):
    help = 'Trains the KNN model on MRI brain image data and saves it to disk.'

    def handle(self, *args, **kwargs):
        healthy_dir = "/Volumes/TOSHIBA EXT/Healthy_Brain_Images"
        tumor_dir = "/Volumes/TOSHIBA EXT/Tumor_Brain_Images"

        self.stdout.write("📦 Loading data...")
        X, y = load_data(healthy_dir, tumor_dir)

        self.stdout.write("🤖 Training model...")
        model = train_knn(X, y)

        self.stdout.write("📊 Evaluating model...")
        accuracy = evaluate_model(model, X, y)

        model_path = os.path.join("trained_knn_model.pkl")
        joblib.dump((model, accuracy), model_path)

        self.stdout.write(self.style.SUCCESS(
            f"✅ Model trained and saved successfully. Accuracy: {accuracy * 100:.2f}%"
        ))

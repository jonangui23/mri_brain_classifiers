from django.core.management.base import BaseCommand
import os
import joblib
from train_model.data_loader import load_data
from train_model.knn_trainer import train_knn
from train_model.evaluate import evaluate_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
        metrics = evaluate_model(model, X, y)
        plot_and_save_graphs(metrics)
        #Remove y_true and y_pred before saving
        metrics_to_save = {
            k:v for k, v in metrics.items() if k not in ('y_true', 'y_pred')
        }
        model_path = os.path.join("train_model","trained_knn_model.joblib")
        joblib.dump((model, metrics_to_save), model_path)

        self.stdout.write(self.style.SUCCESS(
            f"✅ Model trained and saved successfully. Accuracy: {metrics['accuracy']:.2f}%"
        ))

def plot_and_save_graphs(metrics):
    #does the directory exist?
    output_dir = os.path.join("knnapp", "static", "images")
    os.makedirs("knnapp/static/images", exist_ok=True)
    #Path to save plots
    metrics_path = os.path.join(output_dir, "eval_metrics.png")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    #metric bar plot
    names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    scores = [metrics['accuracy'],metrics['precision'],metrics['recall'],metrics['f1_score']]
    plt.figure(figsize=(8,5))
    plt.bar(names, scores, color='skyblue')
    plt.title('Model evaluation Metrics')
    plt.ylabel('Percentage')
    plt.ylim([0,100])
    plt.savefig(metrics_path)
    plt.close()

    #confustion matrix
    cm = confusion_matrix(metrics['y_true'], metrics['y_pred'])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy','Tumor'], yticklabels=['Healthy', 'Tumor'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(cm_path)
    plt.close()

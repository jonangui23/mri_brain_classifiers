from django.shortcuts import render
from train_model.data_loader import load_data
from train_model.knn_trainer import train_knn
from train_model.evaluate import evaluate_model

def index(request):
    print("Index view called")
    healthy_dir = "/Volumes/TOSHIBA EXT/Healthy_Brain_Images"
    tumor_dir = "/Volumes/TOSHIBA EXT/Tumor_Brain_Images"
    
    X, y = load_data(healthy_dir, tumor_dir)
    model = train_knn(X, y)
    accuracy = evaluate_model(model, X, y)

    context = {
        'accuracy': accuracy * 100
    }
    return render(request, 'knn_app/index.html', context)

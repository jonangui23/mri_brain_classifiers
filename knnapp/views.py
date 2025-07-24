from django.shortcuts import render
from train_model.data_loader import load_data
from train_model.knn_trainer import train_knn
from train_model.evaluate import evaluate_model

def index(request):
    healthy_dir = "/Volumes/TOSHIBA EXT/Healthy_Brain_Images"
    tumor_dir = "/Volumes/TOSHIBA EXT/Tumor_Brain_Images"
    
    X, y = load_data(healthy_dir, tumor_dir)
    model = train_knn(X, y)
    accuracy = evaluate_model(model, X, y)

    context = {
        'accuracy': accuracy * 100
    }
    return render(request, 'knn_app/index.html', context)

# knn_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
]

# brain_knn_project/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('knn_app.urls')),
]

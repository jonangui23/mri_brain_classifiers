# knnapp/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_mri, name='upload_mri'),
    path('run_knn/', views.run_knn, name='run_knn'),
]

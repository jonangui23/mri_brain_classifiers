from django.shortcuts import render

# Create your views here.
# knnapp/views.py

from django.shortcuts import render, redirect
from .forms import MRIImageForm
from .models import MRIImage
from .ml.knn_model import train_knn_model, classify_image

def upload_mri(request):
    if request.method == 'POST':
        form = MRIImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('upload_mri')
    else:
        form = MRIImageForm()
    return render(request, 'knnapp/upload.html', {'form': form})

def run_knn(request):
    images = MRIImage.objects.all()
    result = train_knn_model(images)
    return render(request, 'knnapp/result.html', {'result': result})

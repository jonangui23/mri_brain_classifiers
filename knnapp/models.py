from django.db import models

# Create your models here.
# knnapp/models.py

from django.db import models

class MRIImage(models.Model):
    image_file = models.FileField(upload_to='mri_images/')
    label = models.CharField(max_length=20, choices=[('healthy', 'Healthy'), ('tumor', 'Tumor')], blank=True)

    def __str__(self):
        return f"{self.image_file.name} - {self.label}"

from django.db import models

class Detection(models.Model):
    label = models.CharField(max_length=255)
    confidence = models.FloatField()
    bbox = models.CharField(max_length=255)  # Store as a string of coordinates
    timestamp = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='detections/', null=True, blank=True)

    def __str__(self):
        return f"Detection of {self.label} at {self.timestamp}"

class CapturedImage(models.Model):
    image_path = models.CharField(max_length=255)
    annotated_path = models.CharField(max_length=255)
    detections = models.JSONField()  # To store detection data as JSON
    timestamp = models.DateTimeField(auto_now_add=True)
    source = models.CharField(max_length=50, choices=[('capture', 'Capture'), ('upload', 'Upload')], default='upload')


    def __str__(self):
        return f"Image {self.id} - {self.timestamp}"
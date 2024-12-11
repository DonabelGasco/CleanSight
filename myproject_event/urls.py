"""
URL configuration for myproject_event project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
        path('', views.home, name='home'),
        path('live/', views.live, name='live'),
        path('stop_live/', views.stop_live, name='stop_live'),
        path('video_feed/', views.video_feed, name='video_feed'),
        path('capture/', views.capture, name='capture'),
        path('capture/live/', views.capture_live, name='capture_live'),
         path('captured-images/', views.captured_images_list, name='captured_images_list'),
        path('upload/image/', views.upload_image, name='upload_image'),
        path('processed-images/', views.processed_images_list, name='processed_images_list'),
        path('upload/video/', views.upload_video, name='upload_video'),
        path('real_time_detections/', views.real_time_detections, name='real_time_detections'),
        path('delete-image/<int:image_id>/', views.delete_image, name='delete_image'),
        path('static/outputs/<str:filename>/', views.serve_video, name='serve_video'),
]

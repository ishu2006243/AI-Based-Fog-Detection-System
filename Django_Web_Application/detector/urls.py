from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/detect/", views.detect_api, name="detect_api"),
    path("api/serial/status/", views.serial_status_api, name="serial_status_api"),
    path("samples/<str:filename>/", views.sample_image, name="sample_image"),
]

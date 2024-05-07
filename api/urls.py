# urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.my_view),  # Use the main view function for the root URL
]
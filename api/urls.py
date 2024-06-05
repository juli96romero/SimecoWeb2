# urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.my_view),  # Use the main view function for the root URL
    path('api/obj-files/', views.list_obj_files, name='list_obj_files'),
    path('api/save-image/', views.save_image, name='save_image'),
    path('vtk/',views.vtk_visualization, name='vtkVisual'),
    path('api/stl-files/', views.list_stl_files, name='list_stl_files'),
    path('update_visualization', views.update_visualization, name='update_visualization'),
    path('image/', views.vtk_image, name='vtk_visualization'),
    path('fov/', views.pruebaFOV, name='pruebaFov'),
]
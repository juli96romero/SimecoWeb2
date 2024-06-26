# urls.py

from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.main_page, name='api_main_page'),
    path('red128/', views.red128, name='Red 1 inferencia 128px'),
    path('red256/', views.red256, name='Red 2 inferencia 256px'),  
    path('vtk/',views.vtk_visualizador, name='Ver recorte de la malla (ventana_aparte)'),
    path('image/', views.vtk_image, name='Imagen generada con recorte de la malla'),
    #path('fov/', views.pruebaFOV, name='Img output en cartesianas'),
    path('malla/', views.mallaDelFOV, name='FOV del transductor en VTK (ventana_aparte)'),
    path('front/', include('frontend.urls'), name='SimecoWEB'),
    

    ######ENDPOINTS name comienza con 'api_'
    path('api/update_visualization/', views.update_visualization, name='api_update'),
    path('api/stl-files/', views.list_stl_files, name='api_list_stl_files'),
    path('api/obj-files/', views.list_obj_files, name='api_list_obj_files'),
    #path('api/save-image/', views.save_image, name='api_save_image'),
]
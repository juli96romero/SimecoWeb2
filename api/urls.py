# urls.py

from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.main_page, name='api_main_page'),
    path('red128/', views.red128, name='Red 1 inferencia 128px'),
    path('red256/', views.red256, name='Red 2 inferencia 256px'),  
    path('vtk/',views.vtk_visualizador, name='Ver recorte de la malla (ventana_aparte)'),
    path('image/', views.vtk_image, name='Imagen generada con recorte de la malla'),
    path('fov/', views.pruebaFOV, name='Img output en cartesianas'),
    path('malla/', views.mallaDelFOV, name='FOV del transductor en VTK (ventana_aparte)'),
    path('front/', include('frontend.urls'), name='SimecoWEB'),
    path('color/', views.color, name='Img output en cartesianas'),
    path('prueba/', views.pruebaRecorte, name='Prueba'),
    path('prueba2/', views.pruebaRecorte2, name='Prueba2'),
    path('brightness/', views.brightness, name='Red 1 inferencia 128px con brillo'),
    path('poly/', views.createpoly_ellipsoid, name='malla poly para volumen de la panza a partir de ellipsoid'),
    path('polySpline/', views.createpoly_spline, name='Crea malla poly para el volumen de la panza a partir de ParametricSpline'),
    path('movement/', views.vtk_visualization_with_mov, name='Movimiento fallido con la perspectiva'),
    path('movement_ellipsoid/', views.createpoly_ellipsoid_with_mov, name='Movimiento'),


    ######ENDPOINTS name comienza con 'api_'
    path('api/update_visualization/', views.update_visualization, name='api_update'),
    path('api/update_normal/', views.update_normal, name='api_update'),
    path('api/update_position/', views.update_position, name='api_update'),
    path('api/stl-files/', views.list_stl_files, name='api_list_stl_files'),
    path('api/stl-transductor/', views.list_stl_files_transductor, name='api_list_stl_files_transductor'),
    path('api/obj-files/', views.list_obj_files, name='api_list_obj_files'),
    #path('api/save-image/', views.save_image, name='api_save_image'),
]
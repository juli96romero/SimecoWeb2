# urls.py

from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.main_page, name='api_main_page'),
    path('red128/', views.red128, name='Red 1 con 128x128px'),
    path('red256/', views.red256, name='Red 2 con 256x256px'),  
    path('vtk/',views.vtk_visualizador, name='Visualizador de mallas y recorte'),
    path('vtk_mover/',views.interfaz_html, name='VTK con colores y transductor'),
    path('image/', views.vtk_image, name='Imagen generada con recorte'),
    path('solo_la_piel/', views.solo_la_piel, name='Piel + recorte'),
    path('fov/', views.pruebaFOV, name='FOV transductor'),
    path('combined/', views.combinedSlice, name='Combinado'),
    path('brightness/', views.brightness, name='128px con brillo'),
    path('poly/', views.createpoly_ellipsoid, name='Malla poly'),
    path('polySpline/', views.createpoly_spline, name='Malla ParametricSpline'),
    path('movement_ellipsoid/', views.createpoly_ellipsoid_with_mov, name='Movimiento transductor'),

    path('front/', include('frontend.urls'), name='SimecoWEB'),

    ######ENDPOINTS name comienza con 'api_'
    path('api/update_visualization/', views.update_visualization, name='api_update'),
    path('api/mover-transductor/', views.mover_transductor, name='api_update_transducer_pose'),
    path('api/update_normal/', views.update_normal, name='api_update'),
    path('api/update_position/', views.update_position, name='api_update'),
    path('api/stl-files/', views.list_stl_files, name='api_list_stl_files'),
    path('api/stl-transductor/', views.list_stl_files_transductor, name='api_list_stl_files_transductor'),
    path('api/obj-files/', views.list_obj_files, name='api_list_obj_files'),
    #path('api/save-image/', views.save_image, name='api_save_image'),
]
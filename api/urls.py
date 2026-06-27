from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.main_page, name='api_main_page'),
    path('red128/', views.red128, name='Red 1 con 128x128px'),
    path('red256/', views.red256, name='Red 2 con 256x256px'),
    path('vtk/', views.vtk_visualizador, name='Visualizador de mallas y recorte'),
    path('vtk_mover/', views.vtk_mover_page, name='VTK con colores y transductor'),
    path('image/', views.vtk_image, name='Imagen generada con recorte'),
    path('solo_la_piel/', views.skin_only_view, name='Piel + recorte'),
    path('fov/', views.fov_page, name='FOV transductor'),
    path('combined/', views.combined_slice_page, name='Combinado'),
    path('brightness/', views.brightness_page, name='128px con brillo'),
    path('poly/', views.create_poly_ellipsoid, name='Malla poly'),
    path('polySpline/', views.create_poly_spline, name='Malla ParametricSpline'),
    path('movement_ellipsoid/', views.create_poly_ellipsoid_with_movement, name='Movimiento transductor'),
    path('front/', include('frontend.urls'), name='SimecoWEB'),

    path('api/update_visualization/', views.update_visualization, name='api_update_visualization'),
    path('api/mover-transductor/', views.move_transducer, name='api_mover_transducer'),
    path('api/update_normal/', views.update_normal, name='api_update_normal'),
    path('api/update_position/', views.update_position, name='api_update_position'),
    path('api/stl-files/', views.list_stl_files, name='api_list_stl_files'),
]

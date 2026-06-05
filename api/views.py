from django.urls import get_resolver, URLPattern, URLResolver
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import os
import json
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtk.util import numpy_support
import numpy as np
import math
from django.views.decorators.csrf import csrf_exempt

mallas_colors = []
transductor = []
mallas = []
transductor_polydata = None
normal_global = [0.3 , 0.3 , 0.99]
slice_origin = [0, 0, -0.02]
slice_normal = normal_global

mesh_colors = {
    'pelvis': (151/255.0, 151/255.0, 147/255.0),      # gris
    'spleen': (1.0, 0, 1.0),                          # magenta
    'liver': (100/255.0, 0, 100/255.0),               # violeta oscuro
    'surrenalgland': (1.0, 1.0, 0),                   # amarillo
    'kidney': (0, 1.0, 1.0),                          # cian
    'gallbladder': (0, 1.0, 0),                       # verde
    'pancreas': (0, 0, 1.0),                          # azul
    'artery': (0, 0, 1),                              # rojo
    'bones': (1.0, 1.0, 1.0)                          # blanco
}

def red128(request): 
    return render(request, 'api/red128.html')

def red256(request):
    return render(request, 'api/red256.html')

def vtk_image(request):
    return render(request, 'api/vtk_image.html')

def vtk_mover_page(request):
    load_stl_transducer()
    return render(request, 'api/interfaz.html')

def fov_page(request):
    return render(request, 'api/fov.html')

def vtk_visualizador(request):
    return render(request, 'api/vtk_visualizador.html')

def load_stl_transducer():
    global mallas, transductor_polydata

    if mallas and transductor_polydata:
        return  # Ya cargado

    folder_path = "api/stl-files"
    archivos = [f for f in os.listdir(folder_path) if f.endswith('.stl')]

    for archivo in archivos:
        reader = vtk.vtkSTLReader()
        reader.SetFileName(os.path.join(folder_path, archivo))
        reader.Update()
        polydata = reader.GetOutput()

        name = os.path.splitext(archivo)[0].lower()
        if "transductor" in name:
            transductor_polydata = polydata

    load_meshes()

def skin_only_view(request):
    folder_path = os.path.join(os.path.dirname(__file__), "stl-files")
    skin_files = [
        f for f in os.listdir(folder_path)
        if "skin" in f.lower() and f.endswith(".stl")
    ]
    if not skin_files:
        return HttpResponse("No se encontró archivo de piel (.stl)", status=404)

    reader = vtk.vtkSTLReader()
    reader.SetFileName(os.path.join(folder_path, skin_files[0]))
    reader.Update()
    polydata = reader.GetOutput()

    renderer = vtk.vtkRenderer()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(1)
    renderer.AddActor(actor)
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetWindowName("Transductor")
    render_window.SetSize(1000, 800)
    renderer.SetBackground(1, 1, 1)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    render_window.Render()
    interactor.Initialize()
    interactor.Start()
    
    return HttpResponse(f"")

def mostrar_ventana_vtk():
    global ultima_posicion, ultima_rotacion

    renderer = vtk.vtkRenderer()

    for polydata, color in mallas:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(0.2)
        renderer.AddActor(actor)

    if transductor_polydata:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(transductor_polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0, 0)
        actor.GetProperty().SetOpacity(0.6)

        transform = vtk.vtkTransform()
        transform.PostMultiply()
        transform.RotateX(ultima_rotacion[0])
        transform.RotateY(ultima_rotacion[1])
        transform.RotateZ(ultima_rotacion[2])
        transform.Translate(ultima_posicion)
        actor.SetUserTransform(transform)

        renderer.AddActor(actor)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetWindowName("Transductor")
    render_window.SetSize(1000, 800)
    renderer.SetBackground(1, 1, 1)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    render_window.Render()
    interactor.Initialize()
    interactor.Start()

@csrf_exempt
def mover_transductor(request):
    global ultima_posicion, ultima_rotacion

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            ultima_posicion = data.get("position", [0, 0, 0])
            ultima_rotacion = data.get("rotation", [0, 0, 0])
            mostrar_ventana_vtk()
            return JsonResponse({"success": True})
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})
    return JsonResponse({"success": False, "error": "Mtodo invlido"})

#####funcionalidad de htmls:
def extract_patterns(urlpatterns, base=''):
    patterns = []
    for pattern in urlpatterns:
        if isinstance(pattern, URLResolver):
            patterns += extract_patterns(pattern.url_patterns, base + str(pattern.pattern))
        elif isinstance(pattern, URLPattern):
            if pattern.name and not pattern.name.startswith('api_'):  # Excluye nombres que comienzan con 'api_'
                patterns.append({
                    'name': pattern.name,
                    'url': base + str(pattern.pattern)
                })
    return patterns

def main_page(request):
    resolver = get_resolver()
    api_resolver = None
    for pattern in resolver.url_patterns:
        if isinstance(pattern, URLResolver) and pattern.pattern._route == '':
            api_resolver = pattern
            break
    # Extrae los patrones (lista de dict con 'name' y 'url')
    urls = extract_patterns(api_resolver.url_patterns) if api_resolver else []

    # Diccionario de descripciones usando los 'name' de cada URL
    descripciones = {
        'api_main_page': '',  # Página principal sin descripción
        'Red 1 con 128x128px': 'Inferencia con imágenes preexistentes de 128x128 píxeles',
        'Red 2 con 256x256px': 'Inferencia con imágenes preexistentes de 256x256 píxeles',
        'Visualizador de mallas y recorte': 'Visualización de las mallas en VTK junto con el plano generado',
        'VTK con colores y transductor': 'Visualización de las mallas con el transductor y los colores en los órganos',
        'Imagen generada con recorte': 'Generación de imagen a partir del recorte de la malla',
        'Piel + recorte': 'Generación de imagen que muestra la piel junto con el recorte',
        'FOV transductor': 'Transformación al campo de visión (FOV) del transductor',
        'Combinado': 'Generación de imagen combinada con recorte de malla y optimización del FOV. Generado a partir del recorte hecho en VTK',
        '128px con brillo': 'Endpoint para la red de inferencia con imágenes de 128x128 píxeles y ajuste de brillo (imagenes preexistentes)',
        'Malla poly': 'Creación de malla poligonal para el volumen de la panza a partir de un modelo elipsoidal',
        'Malla ParametricSpline': 'Creación de malla poligonal para el volumen de la panza a partir de un modelo spline paramétrico',
        'Movimiento transductor': 'Endpoint para mover el transductor utilizando un modelo de movimiento elipsoidal',
        'SimecoWEB': 'Acceso a la interfaz web de Simeco'
    }
    # Agregar la descripción a cada elemento de urls
    for url in urls:
        url['description'] = descripciones.get(url['name'], 'Descripción no disponible')

    return render(request, 'pagina_principal.html', {'urls': urls})

# funcionalidad de SIMECO#############

#levantar los STL o OBJ
def list_obj_files(request):
    directory_path = os.path.join(os.path.dirname(__file__), 'obj-files')
    files = [f for f in os.listdir(directory_path) if f.endswith('.obj')]
    return JsonResponse(files, safe=False)

def list_stl_files(request):
    directory_path = os.path.join(os.path.dirname(__file__), 'stl-files')
    files = [f for f in os.listdir(directory_path) if f.endswith('.stl')]
    return JsonResponse({'files': files}, safe=False)

def list_transducer_stl_files(request):
    directory_path = os.path.join(os.path.dirname(__file__), 'stl-files')
    files = [f for f in os.listdir(directory_path) if f.endswith('004.stl')]
        
    return JsonResponse({'files': files}, safe=False)

def load_meshes():

    global mallas, transductor

    folder_path = "api/stl-files"
    stl_files = [f for f in os.listdir(folder_path) if f.endswith('.stl')]

    if not stl_files:
        print("No STL files found in the directory.")
        return

    mallas.clear()
    transductor.clear()

    for stl_file in stl_files:
        if "skin" in stl_file.lower():
            print(f"Saltando {stl_file} porque contiene 'skin'")
            continue
        reader = vtk.vtkSTLReader()
        reader.SetFileName(os.path.join(folder_path, stl_file))
        reader.Update()

        polydata = reader.GetOutput()

        # -----------------------------
        # Asignar color según nombre
        # -----------------------------
        name = os.path.splitext(stl_file)[0].lower()
        assigned_color = (0.3, 0.3, 0.3)  # gris por defecto

        for keyword, color in mesh_colors.items():
            if keyword.lower() in name:
                print(f"Asignando color {color} a {stl_file} porque contiene '{keyword}'")
                assigned_color = color
                break

        # -----------------------------
        # Guardamos malla + color juntos
        # -----------------------------
        if stl_file.lower().startswith("transductor"):
            transductor.append((polydata, assigned_color))
        else:
            mallas.append((polydata, assigned_color))


        for keyword, color in mesh_colors.items():
            if keyword in name:
                assigned_color = color
                break
        mallas_colors.append(assigned_color)

    return

def get_meshes():
    if not mallas:
        load_meshes()
    return mallas

def update_visualization(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        normal = data.get('normal', [0.0, 0.0, 1])
        normal_global[0] = normal[0]
        normal_global[1] = normal[1]
        normal_global[2] = normal[2]
        vtk_visualization(request,normal)
        return JsonResponse({'success': True})
    return JsonResponse({'success': False})

@csrf_exempt
def update_normal(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Método inválido'})
    data = json.loads(request.body)
    normal = data.get('normal', [0.0, 0.0, 1])
    normal_global[0] = normal[0]
    normal_global[1] = normal[1]
    normal_global[2] = normal[2]
    return JsonResponse({'success': True})

object_position = [0.9, 0.0, 0.0]  # Initial position

def update_position(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        direction = data.get('direction')
        
        # Adjust the position based on the direction
        if direction == 'left':
            object_position[0] -= 0.1
        elif direction == 'right':
            object_position[0] += 0.1
        elif direction == 'up':
            object_position[1] += 0.1
        elif direction == 'down':
            print("direction")
            object_position[1] -= 0.1

        return JsonResponse({'success': True})
    return JsonResponse({'success': False})

def vtk_visualization(request, normal=normal_global):#RV
    # Create a VTK renderer
    renderer = vtk.vtkRenderer()
    
    # Asegrate de que las mallas estn disponibles
    if not mallas:
        load_meshes()
    print("normal en vtk_visualization:", normal)
    # Agregar las mallas existentes
    for malla, color in mallas:        
        filled_slice = slice_and_fill_mesh_vtk(
            malla, 
            origin=[0, 0, 0], 
            normal=normal
        )
        mesh_mapper = vtk.vtkPolyDataMapper()
        mesh_mapper.SetInputData(malla)
        mesh_actor = vtk.vtkActor()
        mesh_actor.SetMapper(mesh_mapper)
        mesh_actor.GetProperty().SetColor(color)
        mesh_actor.GetProperty().SetOpacity(0.1)
        
        slice_mapper = vtk.vtkPolyDataMapper()
        slice_mapper.SetInputData(filled_slice)
        slice_actor = vtk.vtkActor()
        slice_actor.SetMapper(slice_mapper)
        slice_actor.GetProperty().SetColor(color)
        slice_actor.GetProperty().SetLighting(False)
        
        renderer.AddActor(mesh_actor)
        renderer.AddActor(slice_actor)
    
    transform = vtk.vtkTransform()
    transform.PostMultiply()
    transform.RotateX(-90)
    transform.RotateY(0)
    transform.RotateZ(0)
    transform.Translate(0, 0, -1.25)

    # Configuracion de la ventana de renderizado
    renderer.SetBackground(1, 1, 1)
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1200, 800)
    render_window.SetWindowName("VTK Visualization")

    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Renderizar y empezar la interaccion
    render_window.Render()
    render_window_interactor.Initialize()
    render_window_interactor.Start()

    return True


def slice_and_fill_mesh_vtk(mesh, origin=slice_origin, normal=slice_normal):
    # Create a plane to slice the mesh
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(normal)

    # Create a cutter to get the intersection lines
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(mesh)
    cutter.Update()
    
    # Get the cut lines
    cut_lines = cutter.GetOutput()
    
    # Convert lines to a polygon by using vtkStripper and vtkTriangleFilter
    stripper = vtk.vtkStripper()
    stripper.SetInputData(cut_lines)
    stripper.Update()
    
    # Create a polyline to polygon
    polyline_to_polygon = vtk.vtkPolyData()
    polyline_to_polygon.SetPoints(stripper.GetOutput().GetPoints())
    polyline_to_polygon.SetPolys(stripper.GetOutput().GetLines())
    
    # Triangulate the polygon to fill the slice
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(polyline_to_polygon)
    triangle_filter.Update()
    
    # Get the filled slice
    filled_slice = triangle_filter.GetOutput()
    
    return filled_slice


renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetOffScreenRendering(1) 
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)


def slice_to_image(filled_slices, mesh_colors):
    # Limpiar el renderer de actores anteriores
    renderer.RemoveAllViewProps()

    for i, filled_slice in enumerate(filled_slices):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(filled_slice)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Asignar el color según el indice de la malla
        color = mesh_colors[i]
        actor.GetProperty().SetColor(color)
        actor.GetProperty().LightingOff()  

        renderer.AddActor(actor)

    renderer.SetBackground(0, 0, 0)

    render_window.Render()
    
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.Update()

    vtk_image = window_to_image_filter.GetOutput()
    
    width, height, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    components = vtk_array.GetNumberOfComponents()
    arr = vtk_to_numpy(vtk_array).reshape(height, width, components)
    
    return arr


def vtk_visualization_image(request): #RV
    if not mallas:
        load_meshes()
    data = json.loads(request)
    x = data['x']
    y = data['y']
    z = data['z']
    
    normal_global[0]=x
    normal_global[1]=y
    normal_global[2]=z

    if not mallas:
        load_meshes()
    
    filled_slices = []
    for malla, color in mallas:
        filled_slice = slice_and_fill_mesh_vtk(malla, origin=[0, 0, -0.02], normal=normal_global)
        filled_slices.append(filled_slice)

    slice_image = slice_to_image(filled_slices, mallas_colors)
    
    return slice_image

def combined_slice_page(request):
    return render(request, 'api/combinedSlice.html')

def brightness_page(request):
    return render(request, 'api/brightness.html')


def create_poly_ellipsoid(request):
    folder_path = "api/stl-files/0_skin.stl"
    skin = []
    reader = vtk.vtkSTLReader()
    print(folder_path)
    reader.SetFileName(folder_path)
    reader.Update()
    skin.append(reader.GetOutput())

    # Obtener las dimensiones de la malla de la piel
    skin_poly_data = reader.GetOutput()
    bounds = skin_poly_data.GetBounds()
    print(f"Dimensiones de la malla de la piel: {bounds}")

    # Calcular los radios del elipsoide basados en las dimensiones de la malla
    x_radius = (bounds[1] - bounds[0]) / 2.0  # Radio en X
    y_radius = (bounds[3] - bounds[2]) / 2.0  # Radio en Y
    z_radius = (bounds[5] - bounds[4]) / 2.0  # Radio en Z

    # Aumentar los radios en un 5%
    scale_factor = 1.05  # Aumento del 5%
    x_radius *= scale_factor 
    y_radius *= scale_factor + 0.10
    z_radius *= scale_factor

    print(f"Radios del elipsoide (aumentados en 5%): X={x_radius}, Y={y_radius}, Z={z_radius}")

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    skin_mapper = vtk.vtkPolyDataMapper()
    skin_mapper.SetInputConnection(reader.GetOutputPort())

    skin_actor = vtk.vtkActor()
    skin_actor.SetMapper(skin_mapper)

    # Configurar propiedades de la piel para un aspecto más realista
    skin_property = skin_actor.GetProperty()
    skin_property.SetColor(1.0, 0.8, 0.6)          # Color piel
    skin_property.SetAmbient(0.2)                  # Luz ambiental suave
    skin_property.SetDiffuse(0.8)                  # Reflexión difusa
    skin_property.SetSpecular(0.3)                  # Reflexión especular
    skin_property.SetSpecularPower(10)              # Brillo especular

    # Crear el elipsoide con los radios ajustados
    ellipsoid = vtk.vtkParametricEllipsoid()
    ellipsoid.SetXRadius(x_radius)
    ellipsoid.SetYRadius(y_radius)
    ellipsoid.SetZRadius(z_radius)

    ellipsoid_source = vtk.vtkParametricFunctionSource()
    ellipsoid_source.SetParametricFunction(ellipsoid)
    ellipsoid_source.SetUResolution(50)  # Resolución de la malla del elipsoide
    ellipsoid_source.SetVResolution(50)
    ellipsoid_source.Update()

    ellipsoid_mapper = vtk.vtkPolyDataMapper()
    ellipsoid_mapper.SetInputConnection(ellipsoid_source.GetOutputPort())

    ellipsoid_actor = vtk.vtkActor()
    ellipsoid_actor.SetMapper(ellipsoid_mapper)
    ellipsoid_actor.GetProperty().SetColor(0.5, 0.5, 1)  # Color azul para el elipsoide
    ellipsoid_actor.GetProperty().SetOpacity(0.5)        # Hacerlo semi-transparente

    # Centrar el elipsoide en la misma posición que la malla de la piel
    ellipsoid_actor.SetPosition(
        (bounds[0] + bounds[1]) / 2.0,  # Centro en X
        (bounds[2] + bounds[3]) / 2.0,  # Centro en Y
        (bounds[4] + bounds[5]) / 2.0   # Centro en Z
    )

    # Añadir los actores al renderer
    renderer.AddActor(skin_actor)
    renderer.AddActor(ellipsoid_actor)

    # Configurar la cámara y renderizar
    renderer.ResetCamera()
    render_window.Render()

    # Iniciar el interactor
    interactor.Start()
    return HttpResponse("")

def create_poly_ellipsoid_with_movement(request):
    folder_path = "api/stl-files/0_skin.stl"
    reader = vtk.vtkSTLReader()
    reader.SetFileName(folder_path)
    reader.Update()

    # Obtener las dimensiones de la malla de la piel
    skin_poly_data = reader.GetOutput()
    bounds = skin_poly_data.GetBounds()
    print(f"Dimensiones de la malla de la piel: {bounds}")

    # Calcular los radios del elipsoide basados en las dimensiones de la malla
    x_radius = (bounds[1] - bounds[0]) / 2.0  # Radio en X
    y_radius = (bounds[3] - bounds[2]) / 2.0  # Radio en Y
    z_radius = (bounds[5] - bounds[4]) / 2.0  # Radio en Z

    # Aumentar los radios en un 5%
    scale_factor = 1.05  # Aumento del 5%
    x_radius *= scale_factor
    y_radius *= scale_factor + 0.10
    z_radius *= scale_factor

    print(f"Radios del elipsoide (aumentados en 5%): X={x_radius}, Y={y_radius}, Z={z_radius}")

    # Crear el renderer, la ventana de renderizado y el interactor
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Mapear y mostrar la piel
    skin_mapper = vtk.vtkPolyDataMapper()
    skin_mapper.SetInputConnection(reader.GetOutputPort())

    skin_actor = vtk.vtkActor()
    skin_actor.SetMapper(skin_mapper)
    skin_actor.GetProperty().SetColor(1, 0.5, 0.5)  # Color rosado para la piel
    renderer.AddActor(skin_actor)

    # Crear el elipsoide con los radios ajustados
    ellipsoid = vtk.vtkParametricEllipsoid()
    ellipsoid.SetXRadius(x_radius)
    ellipsoid.SetYRadius(y_radius)
    ellipsoid.SetZRadius(z_radius)

    ellipsoid_source = vtk.vtkParametricFunctionSource()
    ellipsoid_source.SetParametricFunction(ellipsoid)
    ellipsoid_source.SetUResolution(50)  # Resolucin de la malla del elipsoide
    ellipsoid_source.SetVResolution(50)
    ellipsoid_source.Update()

    ellipsoid_mapper = vtk.vtkPolyDataMapper()
    ellipsoid_mapper.SetInputConnection(ellipsoid_source.GetOutputPort())

    ellipsoid_actor = vtk.vtkActor()
    ellipsoid_actor.SetMapper(ellipsoid_mapper)
    ellipsoid_actor.GetProperty().SetColor(0.5, 0.5, 1)  # Color azul para el elipsoide
    ellipsoid_actor.GetProperty().SetOpacity(0.5)  # Hacerlo semi-transparente

    # Centrar el elipsoide en la misma posicin que la malla de la piel
    center_x = (bounds[0] + bounds[1]) / 2.0
    center_y = (bounds[2] + bounds[3]) / 2.0
    center_z = (bounds[4] + bounds[5]) / 2.0
    ellipsoid_actor.SetPosition(center_x, center_y, center_z)
    renderer.AddActor(ellipsoid_actor)

    # Cargar el archivo .stl adicional que se mover alrededor del elipsoide
    moving_stl_path = "api/stl-files/transductor y fov.stl"
    moving_reader = vtk.vtkSTLReader()
    moving_reader.SetFileName(moving_stl_path)
    moving_reader.Update()

    moving_mapper = vtk.vtkPolyDataMapper()
    moving_mapper.SetInputConnection(moving_reader.GetOutputPort())

    moving_actor = vtk.vtkActor()
    moving_actor.SetMapper(moving_mapper)
    moving_actor.GetProperty().SetColor(0, 1, 0)  # Color verde para el objeto mvil
    renderer.AddActor(moving_actor)

    # Variables para controlar el movimiento
    theta = 0  # ngulo en el plano XY (en radianes)
    phi = 0    # ngulo en el plano XZ (en radianes)
    psi = 0    # ngulo en el plano YZ (en radianes)
    delta_angle = 0.02  # Incremento del ngulo por cada evento de teclado
    delta_angle_psi = 0.05  # Incremento del ngulo psi

    # Funcin para actualizar la posicin del objeto mvil
    def update_position():
        nonlocal theta, phi, psi
        # Calcular la posicin en la superficie del elipsoide
        x = center_x + x_radius * cos(theta) * cos(phi)
        y = center_y + y_radius * sin(theta) * cos(psi)
        z = center_z + z_radius * sin(phi) * cos(psi)
        moving_actor.SetPosition(x, y, z)
        print(f"Posicin del objeto mvil: x={x}, y={y}, z={z}")
        render_window.Render()

    # Funcin para manejar el movimiento del objeto
    def move_object(obj, event):
        nonlocal theta, phi, psi
        key = obj.GetKeySym()
        if key == "Left":
            theta -= delta_angle  # Mover en sentido horario en el plano XY
        elif key == "Right":
            theta += delta_angle  # Mover en sentido antihorario en el plano XY
        elif key == "Up":
            phi += delta_angle  # Mover hacia arriba en el plano XZ
        elif key == "Down":
            phi -= delta_angle  # Mover hacia abajo en el plano XZ
        elif key == "a":  # Tecla 'A' para mover en el plano YZ (sentido horario)
            psi -= delta_angle_psi
        elif key == "d":  # Tecla 'D' para mover en el plano YZ (sentido antihorario)
            psi += delta_angle_psi

        # Asegurarse de que los ngulos estn en el rango [0, 2*pi]
        theta = theta % (2 * pi)
        phi = phi % (2 * pi)
        psi = psi % (2 * pi)

        # Actualizar la posicin del objeto mvil
        update_position()

    # Asignar la funcin de manejo de eventos al interactor
    interactor.AddObserver("KeyPressEvent", move_object)

    # Posicionar el objeto mvil en la posicin inicial
    update_position()

    # Configurar la cmara y renderizar
    renderer.ResetCamera()
    render_window.Render()

    # Iniciar el interactor
    interactor.Start()
    return HttpResponse("")


def create_poly_spline(request):
    folder_path = "api/stl-files/skin.stl"
    skin = []
    reader = vtk.vtkSTLReader()
    print(folder_path)
    reader.SetFileName(folder_path)
    reader.Update()
    skin.append(reader.GetOutput())
    print(os.path.exists(reader.GetFileName()))

    # Obtener las dimensiones de la malla de la piel
    skin_poly_data = reader.GetOutput()
    bounds = skin_poly_data.GetBounds()
    print(f"Dimensiones de la malla de la piel: {bounds}")

    # Extraer puntos clave de la malla de la piel
    points = vtk.vtkPoints()
    num_points = skin_poly_data.GetNumberOfPoints()
    for i in range(0, num_points, 100):  # Seleccionar cada 100 puntos (ajusta segn sea necesario)
        point = skin_poly_data.GetPoint(i)
        points.InsertNextPoint(point)

    # Crear la spline a partir de los puntos clave
    spline = vtk.vtkParametricSpline()
    spline.SetPoints(points)

    # Crear una fuente de funcin paramtrica para la spline
    spline_source = vtk.vtkParametricFunctionSource()
    spline_source.SetParametricFunction(spline)
    spline_source.SetUResolution(100)  # Resolucin de la spline
    spline_source.SetVResolution(100)
    spline_source.Update()

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Mapear y crear un actor para la malla de la piel
    skin_mapper = vtk.vtkPolyDataMapper()
    skin_mapper.SetInputConnection(reader.GetOutputPort())

    skin_actor = vtk.vtkActor()
    skin_actor.SetMapper(skin_mapper)
    skin_actor.GetProperty().SetColor(1, 0.5, 0.5)  # Color rosado para la piel

    # Mapear y crear un actor para la spline
    spline_mapper = vtk.vtkPolyDataMapper()
    spline_mapper.SetInputConnection(spline_source.GetOutputPort())

    spline_actor = vtk.vtkActor()
    spline_actor.SetMapper(spline_mapper)
    spline_actor.GetProperty().SetColor(0.5, 0.5, 1)  # Color azul para la spline
    spline_actor.GetProperty().SetOpacity(0.5)  # Hacerlo semi-transparente

    # Aadir los actores al renderer
    renderer.AddActor(skin_actor)
    renderer.AddActor(spline_actor)

    # Configurar la cmara y renderizar
    renderer.ResetCamera()
    render_window.Render()

    # Iniciar el interactor
    interactor.Start()
    return HttpResponse("")


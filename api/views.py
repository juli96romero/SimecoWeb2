from django.urls import get_resolver, URLPattern, URLResolver
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import os
import json
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import numpy as np
import math
from math import cos, sin, pi
from django.views.decorators.csrf import csrf_exempt
from django.urls import get_resolver, URLResolver
from math import cos, sin, pi
from vtk.util import numpy_support

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

def interfaz_html(request):
    levantar_stl()
    return render(request, 'api/interfaz.html')

def pruebaFOV(request):
    return render(request, 'api/fov.html')

def vtk_visualizador(request):
    return render(request, 'api/vtk_visualizador.html')

def vtk_mover(request):
    return render(request, 'api/vtk_visualizador_mover.html')

def levantar_stl():
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

    levantarMallas()

def solo_la_piel(request):

    folder_path = "api/stl_para usar"
    archivos = [f for f in os.listdir(folder_path) if f.endswith('skin.stl')]

    for archivo in archivos:
        reader = vtk.vtkSTLReader()
        reader.SetFileName(os.path.join(folder_path, archivo))
        reader.Update()
        polydata = reader.GetOutput()
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
    radius = 1
    height = 0.1
    angle = math.radians(60)  # Convert degrees to radians
        
    response_data = {
        'files': files,
    }
    return JsonResponse(response_data, safe=False)

def list_stl_files_transductor(request):
    directory_path = os.path.join(os.path.dirname(__file__), 'stl-files')
    files = [f for f in os.listdir(directory_path) if f.endswith('004.stl')]
        
    response_data = {
        'files': files,
    }
    
    return JsonResponse(response_data, safe=False)

#para pasar el fov a STL
def convert_vtk_polydata_to_json(polydata):
    points = polydata.GetPoints()
    num_points = points.GetNumberOfPoints()
    
    data = []
    for i in range(num_points):
        point = points.GetPoint(i)
        data.append({
            'x': point[0],
            'y': point[1],
            'z': point[2]
        })
    
    return data

def apply_color_to_polydata(polydata, color):
    r, g, b = color

    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")

    num_points = polydata.GetNumberOfPoints()

    for _ in range(num_points):
        colors.InsertNextTuple3(
            int(r * 255),
            int(g * 255),
            int(b * 255)
        )

    polydata.GetPointData().SetScalars(colors)

def levantarMallas():

    global mallas, transductor

    folder_path = "api/stl-files"
    stl_files = [f for f in os.listdir(folder_path) if f.endswith('.stl')]

    if not stl_files:
        print("No STL files found in the directory.")
        return

    mallas.clear()
    transductor.clear()

    for stl_file in stl_files:
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

def getMallas():
    if not mallas:
        levantarMallas()
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

def update_normal(request):
    data = json.loads(request)
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
        levantarMallas()
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
        mesh_actor.GetProperty().SetOpacity(0.1)
        
        slice_mapper = vtk.vtkPolyDataMapper()
        slice_mapper.SetInputData(filled_slice)
        slice_actor = vtk.vtkActor()
        slice_actor.SetMapper(slice_mapper)
        slice_actor.GetProperty().SetColor(0.58, 0.0, 0.83)
        
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

###########fovmesh

def create_fov_mesh_with_plane(origin, normal, radius, height, angle):
    # Create the cutting plane using the given origin and normal
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(normal)

    # Create a cone-like shape based on the FOV parameters
    cone = vtk.vtkConeSource()
    cone.SetRadius(radius)
    cone.SetHeight(height)
    cone.SetDirection(normal)  # Align the cone direction with the normal
    cone.SetCenter(origin)
    cone.SetResolution(50)  # Number of segments around the cone

    # Cut the cone with the plane to form the FOV mesh
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputConnection(cone.GetOutputPort())
    cutter.Update()

    # Extract the contour lines from the cut cone
    contour_lines = cutter.GetOutput()

    # Convert contour lines to a polygon (stripper + triangle filter)
    stripper = vtk.vtkStripper()
    stripper.SetInputData(contour_lines)
    stripper.Update()

    # Create a polydata object to store the FOV polygon
    fov_polydata = vtk.vtkPolyData()
    fov_polydata.SetPoints(stripper.GetOutput().GetPoints())
    fov_polydata.SetPolys(stripper.GetOutput().GetLines())

    # Triangulate the polygon to fill the shape
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(fov_polydata)
    triangle_filter.Update()

    # Get the filled FOV mesh
    fov_mesh = triangle_filter.GetOutput()

    return fov_mesh

def create_fov_mesh(origin, normal, radius, height, angle):
    # Normalize the normal vector
    norm = np.linalg.norm(normal)
    normal = [normal[0] / norm, normal[1] / norm, normal[2] / norm]
    
    # Define a default up direction
    up = np.array([0, 0, 1])  # Assuming z-axis is up
    
    # If the normal is parallel or antiparallel to the up vector, choose a different up vector
    if np.allclose(normal, up):
        up = np.array([0, 1, 0])
    elif np.allclose(normal, -up):
        up = np.array([0, -1, 0])
    
    # Create rotation matrix to align up with normal
    v = np.cross(up, normal)
    c = np.dot(up, normal)
    k = 1.0 / (1.0 + c)
    rotation_matrix = np.array([
        [v[0]*v[0]*k + c, v[0]*v[1]*k - v[2], v[0]*v[2]*k + v[1]],
        [v[1]*v[0]*k + v[2], v[1]*v[1]*k + c, v[1]*v[2]*k - v[0]],
        [v[2]*v[0]*k - v[1], v[2]*v[1]*k + v[0], v[2]*v[2]*k + c]
    ])
    
    # Create the points for the FOV shape
    points = vtk.vtkPoints()
    points.InsertNextPoint(origin)
    
    num_segments = 50
    for i in range(num_segments + 1):
        theta = angle * (i / num_segments) - angle / 2
        local_x = radius * math.cos(theta)
        local_y = radius * math.sin(theta)
        local_z = height
        local_point = np.array([local_x, local_y, local_z])
        global_point = origin + np.dot(rotation_matrix, local_point)
        points.InsertNextPoint(global_point.tolist())

    poly = vtk.vtkPolygon()
    poly.GetPointIds().SetNumberOfIds(num_segments + 2)
    for i in range(num_segments + 2):
        poly.GetPointIds().SetId(i, i)
    
    # Create a cell array to store the FOV polygon
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(poly)
    
    # Create a polydata object to store the FOV geometry
    fov_polydata = vtk.vtkPolyData()
    fov_polydata.SetPoints(points)
    fov_polydata.SetPolys(cells)
    
    return fov_polydata


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


def rotation_to_normal(pitch_deg, yaw_deg, roll_deg):
    #Convierte los ángulos de Euler (grados) en un vector normal 3D
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)
    roll = np.radians(roll_deg)

    # Matrices de rotación
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)]
    ])
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll),  np.cos(roll), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx  # rotación combinada
    base_normal = np.array([0.3, 0.3, 0.99])  # tu orientación original
    rotated_normal = R @ base_normal
    rotated_normal /= np.linalg.norm(rotated_normal)
    return rotated_normal.tolist()

def vtk_visualization_image(request): #RV
    if not mallas:
        levantarMallas()
    data = json.loads(request)
    x = data['x']
    y = data['y']
    z = data['z']
    
    normal_global[0]=x
    normal_global[1]=y
    normal_global[2]=z

    if not mallas:
        levantarMallas()
    
    filled_slices = []
    for malla, color in mallas:
        filled_slice = slice_and_fill_mesh_vtk(malla, origin=[0, 0, -0.02], normal=normal_global)
        filled_slices.append(filled_slice)

    slice_image = slice_to_image(filled_slices, mallas_colors)
    
    return slice_image

def combinedSlice(request):
    return render(request, 'api/combinedSlice.html')

def pruebaRecorte2(request):
    return render(request, 'api/pruebaRecorte2.html')

def brightness(request):
    return render(request, 'api/brightness.html')


def generate_cone_mask(image_shape, origin, angle, height):
    mask = np.zeros(image_shape[:2], dtype=bool)
    cx, cy = origin
    for y in range(image_shape[0]):
        for x in range(image_shape[1]):
            dx = x - cx
            dy = y - cy
            distance = math.sqrt(dx**2 + dy**2)
            if distance < (height * math.tan(angle)):
                mask[y, x] = True
    return mask

def apply_fov_to_image(image, mask):
    subimage = np.zeros_like(image)
    subimage[mask] = image[mask]
    return subimage

def generate_subimage_with_fov(request):
    data = json.loads(request)
    x = data['x']
    y = data['y']
    z = data['z']
    
    if not mallas:
        levantarMallas()
    
    slice_image = vtk_visualization_image(request)
    
    origin = (int(slice_image.shape[1] / 2), int(slice_image.shape[0] / 2))
    angle = math.radians(30)  # Ajusta el ngulo del FOV aqu
    height = 100  # Ajusta la altura del FOV aqu
    
    mask = generate_cone_mask(slice_image.shape, origin, angle, height)
    subimage = apply_fov_to_image(slice_image, mask)
    
    mask_image = np.zeros_like(slice_image)
    mask_image[mask] = [255, 255, 255]  # Blanco para la mscara
    
    return slice_image, subimage, mask_image

def createpoly_ellipsoid(request):
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

def createpoly_ellipsoid_with_mov(request):
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


def createpoly_spline(request):
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


def restrict_movement_to_skin(skin_poly_data, transducer_position):
    cell_locator = vtk.vtkCellLocator()
    cell_locator.SetDataSet(skin_poly_data)
    cell_locator.BuildLocator()

    closest_point = [0.0, 0.0, 0.0]
    closest_point_dist2 = vtk.mutable(0.0)
    cell_id = vtk.mutable(0)
    sub_id = vtk.mutable(0)
    cell_locator.FindClosestPoint(transducer_position, closest_point, cell_id, sub_id, closest_point_dist2)

    normals = skin_poly_data.GetPointData().GetNormals()
    if normals:
        normal = [0.0, 0.0, 0.0]
        normals.GetTuple(cell_locator.GetDataSet().FindPoint(closest_point), normal)
    else:
        # Obtener la celda y calcular su normal manualmente
        cell = skin_poly_data.GetCell(cell_id.get())
        if isinstance(cell, vtk.vtkTriangle):
            p0 = np.array(cell.GetPoints().GetPoint(0))
            p1 = np.array(cell.GetPoints().GetPoint(1))
            p2 = np.array(cell.GetPoints().GetPoint(2))
            
            # Calcular el vector normal mediante el producto cruzado
            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2)
            normal /= np.linalg.norm(normal)  # Normalizar el vector
        else:
            normal = [0.0, 0.0, 1.0]  # Normal por defecto en caso de error

    return closest_point, normal

def vtk_visualization_images(mov_module, image_rotation_deg=0):
    global mallas

    if not mallas:
        levantarMallas()

    # -------------------------------------------------
    # Obtener controlador
    # -------------------------------------------------
    controller = mov_module._controller
    if controller is None:
        mov_module.init_controller()
        controller = mov_module._controller

    pos = controller.calculate_position()
    rot = controller.calculate_orientation()

    pos_np = np.array(pos)

    # Centro proyectado al mismo nivel Y
    center_proj = np.array([
        controller.center_x,
        pos_np[1],  # ← misma altura actual
        controller.center_z
    ])

    forward = center_proj - pos_np
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0, 1, 0])
    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0, 0, 1])

    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # -------------------------------------------------
    # Rotaciones locales
    # -------------------------------------------------
    pitch = controller.local_pitch
    yaw   = controller.local_yaw
    roll  = controller.local_roll

    def rot_x(a):
        return np.array([
            [1, 0, 0],
            [0, np.cos(a), -np.sin(a)],
            [0, np.sin(a),  np.cos(a)]
        ])

    def rot_y(a):
        return np.array([
            [ np.cos(a), 0, np.sin(a)],
            [0, 1, 0],
            [-np.sin(a), 0, np.cos(a)]
        ])

    def rot_z(a):
        return np.array([
            [np.cos(a), -np.sin(a), 0],
            [np.sin(a),  np.cos(a), 0],
            [0, 0, 1]
        ])

    R = rot_z(roll) @ rot_y(yaw) @ rot_x(pitch)

    forward = R @ forward
    up      = R @ up

    # -------------------------------------------------
    # Plano de corte
    # -------------------------------------------------
    plane = vtk.vtkPlane()
    plane.SetOrigin(pos)
    plane.SetNormal(up.tolist())

    # -------------------------------------------------
    # Renderer
    # -------------------------------------------------
    ren = vtk.vtkRenderer()
    ren.SetBackground(0, 0, 0)

    actores_creados = 0

    for polydata, color in mallas:

        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(polydata)
        clipper.SetClipFunction(plane)
        clipper.Update()

        filled_poly = clipper.GetOutput()

        if filled_poly.GetNumberOfCells() == 0:
            continue

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(filled_poly)
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetLighting(False)

        ren.AddActor(actor)
        actores_creados += 1

    if actores_creados == 0:
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        return img, pos, rot

    # -------------------------------------------------
    # Bounds combinados
    # -------------------------------------------------
    append_all = vtk.vtkAppendPolyData()

    actors = ren.GetActors()
    actors.InitTraversal()

    for _ in range(actors.GetNumberOfItems()):
        actor = actors.GetNextActor()
        poly = actor.GetMapper().GetInput()
        append_all.AddInputData(poly)

    append_all.Update()
    combined = append_all.GetOutput()
    bounds = combined.GetBounds()

    dx_b = bounds[1] - bounds[0]
    dy_b = bounds[3] - bounds[2]
    dz_b = bounds[5] - bounds[4]

    diagonal = math.sqrt(dx_b*dx_b + dy_b*dy_b + dz_b*dz_b)
    if diagonal < 1e-6:
        diagonal = 1.0

    # -------------------------------------------------
    # Cámara
    # -------------------------------------------------
    renWin = vtk.vtkRenderWindow()
    renWin.SetOffScreenRendering(1)
    renWin.AddRenderer(ren)
    renWin.SetSize(300, 300)

    camera = ren.GetActiveCamera()

    cam_distance = diagonal * 2.0
    cam_pos = pos_np - up * cam_distance

    camera.SetPosition(cam_pos.tolist())
    camera.SetFocalPoint(pos)
    camera.SetViewUp(forward.tolist())

    camera.SetParallelProjection(True)

    scale = max(dx_b, dy_b) * 0.6
    if scale < 1e-3:
        scale = 0.5

    camera.SetParallelScale(scale)

    ren.ResetCameraClippingRange()
    renWin.Render()

    # -------------------------------------------------
    # Capturar imagen
    # -------------------------------------------------
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(renWin)
    w2if.SetInputBufferTypeToRGB()
    w2if.ReadFrontBufferOff()
    w2if.Update()

    vtk_image = w2if.GetOutput()
    dims = vtk_image.GetDimensions()

    vtk_array = vtk_image.GetPointData().GetScalars()
    np_array = numpy_support.vtk_to_numpy(vtk_array)

    img = np_array.reshape(dims[1], dims[0], 3)
    img = np.flipud(img)

    return img.astype(np.uint8), pos, rot


def visualize_cutting_plane(mov_module, plane_size=None):

    # -------------------------------------------------
    # Asegurar que las mallas están cargadas
    # -------------------------------------------------
    global mallas
    if not mallas:
        levantarMallas()

    controller = mov_module._controller
    if controller is None:
        mov_module.init_controller()
        controller = mov_module._controller

    pos = controller.calculate_position()
    pos_np = np.array(pos)

    # Centro proyectado al mismo Y actual (modo cilindro vertical)
    center_proj = np.array([
        controller.center_x,
        pos_np[1],  # misma altura actual
        controller.center_z
    ])

    forward0 = center_proj - pos_np
    forward0 = forward0 / np.linalg.norm(forward0)

    world_up = np.array([0, 1, 0])
    if abs(np.dot(forward0, world_up)) > 0.99:
        world_up = np.array([0, 0, 1])

    right0 = np.cross(forward0, world_up)
    right0 = right0 / np.linalg.norm(right0)

    up0 = np.cross(right0, forward0)
    up0 = up0 / np.linalg.norm(up0)

    pitch = controller.local_pitch
    yaw   = controller.local_yaw
    roll  = controller.local_roll

    def rot_x(a):
        return np.array([
            [1, 0, 0],
            [0, np.cos(a), -np.sin(a)],
            [0, np.sin(a),  np.cos(a)]
        ])

    def rot_y(a):
        return np.array([
            [ np.cos(a), 0, np.sin(a)],
            [0, 1, 0],
            [-np.sin(a), 0, np.cos(a)]
        ])

    def rot_z(a):
        return np.array([
            [np.cos(a), -np.sin(a), 0],
            [np.sin(a),  np.cos(a), 0],
            [0, 0, 1]
        ])

    R = rot_z(roll) @ rot_y(yaw) @ rot_x(pitch)

    # Aplicar rotaciones a los tres vectores base
    forward = R @ forward0
    right   = R @ right0
    up      = R @ up0


    ren = vtk.vtkRenderer()
    ren.SetBackground(0.1, 0.2, 0.4)

    # Añadir las mallas con un poco de transparencia
    for polydata, color in mallas:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(0.7)
        ren.AddActor(actor)

    append_all = vtk.vtkAppendPolyData()
    for polydata, _ in mallas:
        append_all.AddInputData(polydata)
    append_all.Update()
    combined = append_all.GetOutput()
    bounds = combined.GetBounds()
    if bounds[1] < bounds[0]:  # no hay mallas
        diagonal = 10.0
    else:
        dx = bounds[1] - bounds[0]
        dy = bounds[3] - bounds[2]
        dz = bounds[5] - bounds[4]
        diagonal = np.sqrt(dx*dx + dy*dy + dz*dz)
        if diagonal < 1e-6:
            diagonal = 10.0

    if plane_size is None:
        plane_size = diagonal * 0.8

    plane_source = vtk.vtkPlaneSource()
    plane_source.SetOrigin(-0.5, -0.5, 0.0)
    plane_source.SetPoint1( 0.5, -0.5, 0.0)
    plane_source.SetPoint2(-0.5,  0.5, 0.0)

    # Matriz de transformación: alinea X con right, Y con forward, Z con up
    rot_matrix = np.column_stack((right, forward, up))

    transform = vtk.vtkTransform()
    transform.Translate(pos)
    m = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            m.SetElement(i, j, rot_matrix[i, j])
    transform.SetMatrix(m)
    transform.Scale(plane_size, plane_size, 1.0)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputConnection(plane_source.GetOutputPort())
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    plane_polydata = transform_filter.GetOutput()

    mapper_plane = vtk.vtkPolyDataMapper()
    mapper_plane.SetInputData(plane_polydata)
    plane_actor = vtk.vtkActor()
    plane_actor.SetMapper(mapper_plane)
    plane_actor.GetProperty().SetColor(1, 1, 0)       # amarillo
    plane_actor.GetProperty().SetOpacity(0.5)
    plane_actor.GetProperty().SetLighting(False)
    ren.AddActor(plane_actor)


    arrow_source = vtk.vtkArrowSource()
    arrow_source.SetTipResolution(6)
    arrow_source.SetShaftResolution(6)

    arrow_transform = vtk.vtkTransform()
    arrow_transform.Translate(pos)

    x_axis = np.array([1, 0, 0])
    if not np.allclose(up, x_axis):
        v = np.cross(x_axis, up)
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-6:
            v = v / v_norm
            angle = np.arccos(np.clip(np.dot(x_axis, up), -1, 1)) * 180 / np.pi
            arrow_transform.RotateWXYZ(angle, v[0], v[1], v[2])

    arrow_transform.Scale(0.2 * plane_size, 0.2 * plane_size, 0.2 * plane_size)

    arrow_filter = vtk.vtkTransformPolyDataFilter()
    arrow_filter.SetInputConnection(arrow_source.GetOutputPort())
    arrow_filter.SetTransform(arrow_transform)
    arrow_filter.Update()

    mapper_arrow = vtk.vtkPolyDataMapper()
    mapper_arrow.SetInputData(arrow_filter.GetOutput())
    arrow_actor = vtk.vtkActor()
    arrow_actor.SetMapper(mapper_arrow)
    arrow_actor.GetProperty().SetColor(1, 1, 0)   # amarillo
    arrow_actor.GetProperty().SetOpacity(0.3)
    arrow_actor.GetProperty().SetLighting(False)
    ren.AddActor(arrow_actor)


    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetRadius(0.1 * plane_size)  # tamaño relativo
    sphere_source.SetThetaResolution(16)
    sphere_source.SetPhiResolution(16)

    sphere_mapper = vtk.vtkPolyDataMapper()
    sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())

    sphere_actor = vtk.vtkActor()
    sphere_actor.SetMapper(sphere_mapper)
    sphere_actor.SetPosition(pos)
    sphere_actor.GetProperty().SetColor(1, 0, 0)  # rojo
    sphere_actor.GetProperty().SetOpacity(0.4)
    sphere_actor.GetProperty().SetLighting(False)
    ren.AddActor(sphere_actor)


    arrow_forward_source = vtk.vtkArrowSource()
    arrow_forward_source.SetTipResolution(6)
    arrow_forward_source.SetShaftResolution(6)

    # Transformación para la flecha: origen en pos, dirección = forward, longitud = plane_size * 0.5
    arrow_forward_transform = vtk.vtkTransform()
    arrow_forward_transform.Translate(pos)

    # Orientar el eje X de la flecha (por defecto) hacia forward
    if not np.allclose(forward, x_axis):
        v = np.cross(x_axis, forward)
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-6:
            v = v / v_norm
            angle = np.arccos(np.clip(np.dot(x_axis, forward), -1, 1)) * 180 / np.pi
            arrow_forward_transform.RotateWXYZ(angle, v[0], v[1], v[2])

    arrow_forward_transform.Scale(plane_size * 0.5, plane_size * 0.1, plane_size * 0.1)

    arrow_forward_filter = vtk.vtkTransformPolyDataFilter()
    arrow_forward_filter.SetInputConnection(arrow_forward_source.GetOutputPort())
    arrow_forward_filter.SetTransform(arrow_forward_transform)
    arrow_forward_filter.Update()

    mapper_forward_arrow = vtk.vtkPolyDataMapper()
    mapper_forward_arrow.SetInputData(arrow_forward_filter.GetOutput())

    forward_arrow_actor = vtk.vtkActor()
    forward_arrow_actor.SetMapper(mapper_forward_arrow)
    forward_arrow_actor.GetProperty().SetColor(1, 1, 0)  # amarillo para distinguir
    forward_arrow_actor.GetProperty().SetLighting(False)
    ren.AddActor(forward_arrow_actor)


    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(800, 600)
    renWin.SetWindowName("Plano de corte - Visualización 3D (con transductor)")

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)

    ren.ResetCamera()
    renWin.Render()
    iren.Start()
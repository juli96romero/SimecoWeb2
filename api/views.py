from django.urls import get_resolver, URLPattern, URLResolver
from django.shortcuts import render
from django.http import HttpResponse
from .red import main  # Corrected import statement
from django.http import JsonResponse
import os
import json
import base64
from django.conf import settings
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
from io import BytesIO
from PIL import Image
from os import path
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import math
from math import cos, sin, pi
from django.views.decorators.csrf import csrf_exempt
from . import debug_transducer_vtk

mallas_colors = []
transductor = []

# Variables globales para el estado del transductor
theta = 0  # ngulo en el plano XY (en radianes)
phi = 0    # ngulo en el plano XZ (en radianes)
psi = 0    # ngulo en el plano YZ (en radianes)
delta_angle = 0.02  # Incremento del ngulo por cada evento de teclado
delta_angle_psi = 0.05  # Incremento del ngulo psi

# Radios del elipsoide (ajustados segn la malla de la piel)
x_radius = 10.0
y_radius = 15.0
z_radius = 8.0

# Centro del elipsoide
center_x = 0.0
center_y = 0.0
center_z = 0.0

mallas = []
transductor_polydata = None
ultima_posicion = [0, 0, 0]
ultima_rotacion = [0, 0, 0]

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
        else:
            color = (1, 1, 1)
            for palabra, c in mesh_colors.items():
                if palabra in name:
                    color = c
                    break
            mallas.append((polydata, color))

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

def interfaz_html(request):
    levantar_stl()
    return HttpResponse(f"""
        <html>
        <head>
            <title>Control del Transductor</title>
        </head>
        <body style="font-family:sans-serif; text-align:center; padding:50px;">
            <h1>Control del Transductor</h1>
            <label>Posicin (x, y, z):</label><br>
            <input id="posX" type="number" value="0"> 
            <input id="posY" type="number" value="0"> 
            <input id="posZ" type="number" value="0"><br><br>

            <label>Rotacin (x, y, z):</label><br>
            <input id="rotX" type="number" value="0"> 
            <input id="rotY" type="number" value="0"> 
            <input id="rotZ" type="number" value="0"><br><br>

            <button onclick="enviar()">Actualizar vista VTK</button>

            <script>
                function enviar() {{
                    const pos = [
                        parseFloat(document.getElementById("posX").value),
                        parseFloat(document.getElementById("posY").value),
                        parseFloat(document.getElementById("posZ").value)
                    ];
                    const rot = [
                        parseFloat(document.getElementById("rotX").value),
                        parseFloat(document.getElementById("rotY").value),
                        parseFloat(document.getElementById("rotZ").value)
                    ];
                    fetch("/mover-transductor/", {{
                        method: "POST",
                        headers: {{
                            "Content-Type": "application/json"
                        }},
                        body: JSON.stringify({{ position: pos, rotation: rot }})
                    }}).then(resp => resp.json()).then(data => {{
                        if (data.success) {{
                            alert("Ventana actualizada!");
                        }} else {{
                            alert("Error: " + data.error);
                        }}
                    }});
                }}
            </script>
        </body>
        </html>
    """)

mesh_colors = {
    'pelvis': (151/255.0, 151/255.0, 147/255.0),
    'spleen': (1.0, 0, 1.0),
    'liver': (100/255.0, 0, 100/255.0),
    'surrenalGland': (0, 1.0, 1.0),
    'kidney': (1.0, 1.0, 0),
    'gallbladder': (0, 1.0, 0),
    'pancreas': (0, 0, 1.0),
    'artery': (1.0, 0, 0),
    'bones': (1.0, 1.0, 1.0)
}

normal_global = [0.3 , 0.3 , 0.99]
slice_origin = [0, 0, -0.02]
slice_normal = normal_global

#####funcionalidad de htmls:
def extract_patterns(urlpatterns, base=''):
    """Recursively extract all named URL patterns"""
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
    # Encuentra el resolver que corresponde a la aplicacin "api"
    api_resolver = None
    for pattern in resolver.url_patterns:
        if isinstance(pattern, URLResolver) and pattern.pattern._route == '':
            api_resolver = pattern
            break

    urls = extract_patterns(api_resolver.url_patterns) if api_resolver else []
    
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
    print("responsedata", response_data)
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

def move_transducer(action):
    global theta, phi, psi, delta_angle, delta_angle_psi
    global center_x, center_y, center_z
    global x_radius, y_radius, z_radius
    print("valor actual variables globales:", 
          f"theta={theta}, phi={phi}, psi={psi}, "
          f"x_radius={x_radius}, y_radius={y_radius}, z_radius={z_radius}")
    # Actualizar los ngulos segn la accin
    if action == 'left':
        theta -= delta_angle
    elif action == 'right':
        theta += delta_angle
    elif action == 'up':
        phi += delta_angle
    elif action == 'down':
        phi -= delta_angle
    elif action == 'a':
        psi -= delta_angle_psi
    elif action == 'd':
        psi += delta_angle_psi

    # Asegurarse de que los ngulos estn en el rango [0, 2*pi]
    theta = theta % (2 * pi)
    phi = phi % (2 * pi)
    psi = psi % (2 * pi)

    # Calcular la nueva posicin en la superficie del elipsoide
    # (using exactly the same formula as in the first function)
    x = center_x + x_radius * cos(theta) * cos(phi)
    y = center_y + y_radius * sin(theta) * cos(psi)
    z = center_z + z_radius * sin(phi) * cos(psi)

    print(f"Posicin del transductor: x={x:.6f}, y={y:.6f}, z={z:.6f}")
    return (x, y, z)


def levantarMallas():
    folder_path = "api/stl-files"
    stl_files = [f for f in os.listdir(folder_path) if f.endswith('.stl')]

    if not stl_files:
        print("No STL files found in the directory.")
        return

    for stl_file in stl_files:
        reader = vtk.vtkSTLReader()
        reader.SetFileName(os.path.join(folder_path, stl_file))
        reader.Update()
        if stl_file.startswith("transductor"):
            transductor.append(reader.GetOutput())
        else:
            mallas.append(reader.GetOutput())

        # Asignar color segn el nombre del archivo
        name = os.path.splitext(stl_file)[0].lower()
        assigned_color = (1.0, 1.0, 1.0)  # Color por defecto (blanco)
        for keyword, color in mesh_colors.items():
            if keyword in name:
                assigned_color = color
                break
        mallas_colors.append(assigned_color)

    return

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
    #mallaDelFOV(request)
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


def vtk_visualization(request, normal=normal_global):
    # Create a VTK renderer
    renderer = vtk.vtkRenderer()
    
    # Asegrate de que las mallas estn disponibles
    if not mallas:
        levantarMallas()
    print("normal en vtk_visualization:", normal)
    # Agregar las mallas existentes
    for malla in mallas:        
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
    
    # Configurar transductor
    transductor_mapper = vtk.vtkPolyDataMapper()
    transductor_mapper.SetInputData(transductor[0])

    transductor_actor = vtk.vtkActor()
    transductor_actor.SetMapper(transductor_mapper)
    transductor_actor.GetProperty().SetOpacity(0.4)
    transductor_actor.GetProperty().SetColor(1, 0.0, 0.0)
    transform = vtk.vtkTransform()
    transform.PostMultiply()
    transform.RotateX(-90)
    transform.RotateY(0)
    transform.RotateZ(0)
    transform.Translate(0, 0, -1.25)

    # Aplicar transform
    transductor_actor.SetUserTransform(transform)

    renderer.AddActor(transductor_actor)
    

    # Configuracin de la ventana de renderizado
    renderer.SetBackground(1, 1, 1)
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1200, 800)
    render_window.SetWindowName("VTK Visualization")

    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Renderizar y empezar la interaccin
    render_window.Render()
    render_window_interactor.Initialize()
    render_window_interactor.Start()

    return True

####SLICE AND FILL




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

def mallaDelFOV(request):
    # Set up VTK rendering
    renderer = vtk.vtkRenderer()

    if not mallas:
        levantarMallas()
    for malla in mallas:
        liver_mesh = malla
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(liver_mesh)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.1)
        
        renderer.AddActor(actor)
    
    # Define FOV shape parameters
    radius = 1
    height = 0.1
    angle = math.radians(60)  # Convert degrees to radians

    # Create FOV mask on the same plane as slice origin and normal
    fov_mesh = create_fov_mesh(slice_origin, slice_normal, radius, height, angle)

    # Mapper and actor for the FOV mesh
    fov_mapper = vtk.vtkPolyDataMapper()
    fov_mapper.SetInputData(fov_mesh)
    fov_actor = vtk.vtkActor()
    fov_actor.SetMapper(fov_mapper)
    fov_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Set color to red
    fov_actor.GetProperty().SetOpacity(1)
    print(fov_actor.GetPosition())
    fov_actor.SetPosition(0.9,0.0,0.0)
    renderer.SetBackground(1, 1, 1)
    renderer.AddActor(fov_actor)

    # Create a render window and interactor
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Render and interact
    render_window.Render()
    render_window_interactor.Start()
    return render(request, 'api/malla.html')






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
        
        # Asignar el color segn el ndice de la malla
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
    """Convierte los ángulos de Euler (grados) en un vector normal 3D"""
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

def vtk_visualization_image(request, mov):
    data = json.loads(request)

    # --- Obtener la rotación actual del transductor ---
    pos, rot = mov.get_current_position(), mov.get_current_orientation()

    # Convertir la rotación (pitch, yaw, roll) a vector normal
    normal = rotation_to_normal(*rot)

    # --- CORRECCIÓN: asegurar que el plano corte el cuerpo ---
    # Si el transductor está fuera, desplazamos el plano hacia adentro (por el vector normal)
    # offset ajusta qué tan "profundo" entra el plano en el volumen
    offset = 1.05  # este valor suele ser ~la distancia del transductor al centro
    origin = [
        pos[0] + normal[0] * offset,
        pos[1] + normal[1] * offset,
        pos[2] + normal[2] * offset - 0.02  # leve ajuste como tu plano original
    ]


    debug_transducer_vtk.debug_transducer_vtk(
        position=pos,
        orientation=rot,     # pasamos la matriz
        normal=normal,
        target_origin=origin,
        show_skin=True,
        show_position=True,
        show_normal=True,
        show_plane=True,
        show_target=True,
        show_orientation=True,
        show_cone=True,
        scale_factor=0.02
    )
    # --- (opcional) log para depuración ---
    print(f"Posición: {pos}")
    print(f"Rotación (pitch, yaw, roll): {rot}")
    print(f"Normal: {normal}")
    print(f"Origen del plano: {origin}")

    # --- Mantener compatibilidad con tu versión previa ---
    if not mallas:
        levantarMallas()

    filled_slices = []
    for malla in mallas:
        filled_slice = slice_and_fill_mesh_vtk(
            malla,
            origin=origin,
            normal=normal
        )
        filled_slices.append(filled_slice)

    slice_image = slice_to_image(filled_slices, mallas_colors)

    return slice_image, pos, normal


def red128(request):
    return render(request, 'api/red128.html')

def red256(request):
    return render(request, 'api/red256.html')

def vtk_image(request):
    return render(request, 'api/vtk_image.html')

def pruebaFOV(request):
    return render(request, 'api/fov.html')

def vtk_visualizador(request):
    return render(request, 'api/vtk_visualizador.html')

def vtk_mover(request):
    return render(request, 'api/vtk_visualizador_mover.html')



def pruebaRecorte(request):
    return render(request, 'api/pruebaRecorte.html')

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
    print(os.path.exists(reader.GetFileName()))

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
    skin_actor.GetProperty().SetColor(1, 0.5, 0.5)  # Color rosado para la piel

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
    ellipsoid_actor.SetPosition(
        (bounds[0] + bounds[1]) / 2.0,  # Centro en X
        (bounds[2] + bounds[3]) / 2.0,  # Centro en Y
        (bounds[4] + bounds[5]) / 2.0   # Centro en Z
    )

    # Aadir los actores al renderer
    renderer.AddActor(skin_actor)
    renderer.AddActor(ellipsoid_actor)

    # Configurar la cmara y renderizar
    renderer.ResetCamera()
    render_window.Render()

    # Iniciar el interactor
    interactor.Start()
    return HttpResponse("")

import vtk
from math import cos, sin, pi
from django.http import HttpResponse

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

import vtk

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

def vtk_visualization_with_mov(request, normal=normal_global):
    # Crear un renderer de VTK
    renderer = vtk.vtkRenderer()

    # Asegrate de que las mallas estn disponibles
    if not mallas:
        levantarMallas()

    # Agregar las mallas existentes
    for malla in mallas:
        filled_slice = slice_and_fill_mesh_vtk(
            malla,
            origin=[0, 0, -0.02],
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

    # Configurar el transductor
    transductor_mapper = vtk.vtkPolyDataMapper()
    transductor_mapper.SetInputData(transductor[0])

    transductor_actor = vtk.vtkActor()
    transductor_actor.SetMapper(transductor_mapper)
    transductor_actor.GetProperty().SetOpacity(0.4)
    transductor_actor.GetProperty().SetColor(1, 0.0, 0.0)

    # Definir la posicin inicial del transductor
    transductor_position = [0.5, 0.5, 0.5]  # Posicin de ejemplo
    transductor_actor.SetPosition(*transductor_position)
    renderer.AddActor(transductor_actor)

    # Funcin para mover el transductor
    def move_transducer(position):
        valid_position, normal = restrict_movement_to_skin(mallas[0], position)  # Usar la primera malla como piel
        transductor_actor.SetPosition(valid_position)
        # Ajustar la orientacin del transductor (opcional)
        # transductor_actor.SetOrientation(normal)
        render_window.Render()

    # Funcin para manejar eventos del teclado
    def on_key_press(obj, event):
        key = obj.GetKeySym()
        delta = 0.1  # Cantidad de movimiento en cada paso

        if key == "Up":
            transductor_position[1] += delta  # Mover en Y positivo
        elif key == "Down":
            transductor_position[1] -= delta  # Mover en Y negativo
        elif key == "Left":
            transductor_position[0] -= delta  # Mover en X negativo
        elif key == "Right":
            transductor_position[0] += delta  # Mover en X positivo
        elif key == "1":
            transductor_position[2] += delta  # Mover en Z positivo
        elif key == "3":
            transductor_position[2] -= delta  # Mover en Z negativo

        # Restringir el movimiento del transductor
        move_transducer(transductor_position)

    # Configuracin de la ventana de renderizado
    renderer.SetBackground(1, 1, 1)
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1200, 800)
    render_window.SetWindowName("VTK Visualization")

    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Asignar la funcin de manejo de eventos al interactor
    render_window_interactor.AddObserver("KeyPressEvent", on_key_press)

    # Renderizar y empezar la interaccin
    render_window.Render()
    render_window_interactor.Initialize()
    render_window_interactor.Start()

    return True

def color(request):
    from PIL import Image

    # Cargar la imagen
    image_path = "C:/Users/Juli/Desktop/pickleada.png"
    img = Image.open(image_path)
    img = img.convert("RGB")  # Asegurarse de que la imagen est en modo RGB

    # Obtener los pxeles de la imagen
    pixels = img.load()

    # Iterar sobre cada pxel y cambiar el color si no es rojo
    for i in range(img.width):
        for j in range(img.height):
            r, g, b = pixels[i, j]
            if not (r == 255 and g == 0 and b == 0):  # Comprobar si el pxel no es rojo (RGB: (255, 0, 0))
                pixels[i, j] = (255, 255, 255)  # Cambiar el pxel a blanco (RGB: (255, 255, 255))

    # Guardar la imagen modificada
    img.save("C:/Users/Juli/Desktop/pickleada.png")
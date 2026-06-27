from django.urls import get_resolver, URLPattern, URLResolver
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import os
import json
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import numpy as np
import math
from django.views.decorators.csrf import csrf_exempt

mesh_color_list = []
transducer = []
meshes = []
transducer_polydata = None
normal_global = [0.3 , 0.3 , 0.99]
slice_origin = [0, 0, -0.02]
slice_normal = normal_global

mesh_colors = {
    'pelvis': (151/255.0, 151/255.0, 147/255.0),      # gray
    'spleen': (1.0, 0, 1.0),                          # magenta
    'liver': (100/255.0, 0, 100/255.0),               # dark violet
    'surrenalgland': (0, 1.0, 1.0),                   # cyan
    'kidney': (1.0, 1.0, 0),                          # yellow
    'gallbladder': (0, 1.0, 0),                       # green
    'pancreas': (0, 0, 1.0),                          # blue
    'artery': (1, 0, 0),                              # red
    'bones': (1.0, 1.0, 1.0)                          # white
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
    global meshes, transducer_polydata

    if meshes and transducer_polydata:
        return  # already loaded

    folder_path = "api/stl-files"
    files = [f for f in os.listdir(folder_path) if f.endswith('.stl')]

    for file in files:
        reader = vtk.vtkSTLReader()
        reader.SetFileName(os.path.join(folder_path, file))
        reader.Update()
        polydata = reader.GetOutput()

        name = os.path.splitext(file)[0].lower()
        if "transductor" in name:
            transducer_polydata = polydata

    load_meshes()

def skin_only_view(request):
    folder_path = os.path.join(os.path.dirname(__file__), "stl-files")
    skin_files = [
        f for f in os.listdir(folder_path)
        if "skin" in f.lower() and f.endswith(".stl")
    ]
    if not skin_files:
        return HttpResponse("No skin file (.stl) found", status=404)

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

def show_vtk_window():
    global last_position, last_rotation

    renderer = vtk.vtkRenderer()

    for polydata, color in meshes:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(0.2)
        renderer.AddActor(actor)

    if transducer_polydata:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(transducer_polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0, 0)
        actor.GetProperty().SetOpacity(0.6)

        transform = vtk.vtkTransform()
        transform.PostMultiply()
        transform.RotateX(last_rotation[0])
        transform.RotateY(last_rotation[1])
        transform.RotateZ(last_rotation[2])
        transform.Translate(last_position)
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
def move_transducer(request):
    global last_position, last_rotation

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            last_position = data.get("position", [0, 0, 0])
            last_rotation = data.get("rotation", [0, 0, 0])
            show_vtk_window()
            return JsonResponse({"success": True})
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})
    return JsonResponse({"success": False, "error": "Invalid method"})

def extract_patterns(urlpatterns, base=''):
    patterns = []
    for pattern in urlpatterns:
        if isinstance(pattern, URLResolver):
            patterns += extract_patterns(pattern.url_patterns, base + str(pattern.pattern))
        elif isinstance(pattern, URLPattern):
            if pattern.name and not pattern.name.startswith('api_'):  # skip names starting with 'api_'
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
    urls = extract_patterns(api_resolver.url_patterns) if api_resolver else []

    # descriptions keyed by each URL 'name' (UI text in Spanish)
    descriptions = {
        'api_main_page': '',
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
    for url in urls:
        url['description'] = descriptions.get(url['name'], 'Descripción no disponible')

    return render(request, 'pagina_principal.html', {'urls': urls})

def list_stl_files(request):
    directory_path = os.path.join(os.path.dirname(__file__), 'stl-files')
    files = [f for f in os.listdir(directory_path) if f.endswith('.stl')]
    return JsonResponse({'files': files}, safe=False)

def load_meshes():

    global meshes, transducer

    folder_path = "api/stl-files"
    stl_files = [f for f in os.listdir(folder_path) if f.endswith('.stl')]

    if not stl_files:
        print("No STL files found in the directory.")
        return

    meshes.clear()
    transducer.clear()
    mesh_color_list.clear()

    for stl_file in stl_files:
        if "skin" in stl_file.lower():
            print(f"Skipping {stl_file} because it contains 'skin'")
            continue
        reader = vtk.vtkSTLReader()
        reader.SetFileName(os.path.join(folder_path, stl_file))
        reader.Update()

        polydata = reader.GetOutput()

        # assign color by name
        name = os.path.splitext(stl_file)[0].lower()
        assigned_color = (0.3, 0.3, 0.3)  # default gray

        for keyword, color in mesh_colors.items():
            if keyword.lower() in name:
                print(f"Assigning color {color} to {stl_file} because it contains '{keyword}'")
                assigned_color = color
                break

        # store mesh + color together
        if stl_file.lower().startswith("transductor"):
            transducer.append((polydata, assigned_color))
        else:
            meshes.append((polydata, assigned_color))

        mesh_color_list.append(assigned_color)

    return

def get_meshes():
    if not meshes:
        load_meshes()
    return meshes

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
        return JsonResponse({'success': False, 'error': 'Invalid method'})
    data = json.loads(request.body)
    normal = data.get('normal', [0.0, 0.0, 1])
    normal_global[0] = normal[0]
    normal_global[1] = normal[1]
    normal_global[2] = normal[2]
    return JsonResponse({'success': True})

object_position = [0.9, 0.0, 0.0]  # initial position

def update_position(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        direction = data.get('direction')
        
        # adjust the position based on the direction
        if direction == 'left':
            object_position[0] -= 0.1
        elif direction == 'right':
            object_position[0] += 0.1
        elif direction == 'up':
            object_position[1] += 0.1
        elif direction == 'down':
            object_position[1] -= 0.1

        return JsonResponse({'success': True})
    return JsonResponse({'success': False})

def vtk_visualization(request, normal=normal_global):
    renderer = vtk.vtkRenderer()
    
    if not meshes:
        load_meshes()
    print("normal in vtk_visualization:", normal)
    for mesh, color in meshes:        
        filled_slice = slice_and_fill_mesh_vtk(
            mesh, 
            origin=[0, 0, 0], 
            normal=normal
        )
        mesh_mapper = vtk.vtkPolyDataMapper()
        mesh_mapper.SetInputData(mesh)
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

    renderer.SetBackground(1, 1, 1)
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1200, 800)
    render_window.SetWindowName("VTK Visualization")

    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

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


def slice_to_image(filled_slices, colors):
    # clear previous actors from the renderer
    renderer.RemoveAllViewProps()

    for i, filled_slice in enumerate(filled_slices):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(filled_slice)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # assign the color by mesh index
        color = colors[i]
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


def vtk_visualization_image(payload):
    if not meshes:
        load_meshes()
    data = json.loads(payload)
    x = data['x']
    y = data['y']
    z = data['z']
    
    normal_global[0]=x
    normal_global[1]=y
    normal_global[2]=z

    filled_slices = []
    for mesh, color in meshes:
        filled_slice = slice_and_fill_mesh_vtk(mesh, origin=[0, 0, -0.02], normal=normal_global)
        filled_slices.append(filled_slice)

    slice_image = slice_to_image(filled_slices, mesh_color_list)
    
    return slice_image

def combined_slice_page(request):
    return render(request, 'api/combinedSlice.html')

def brightness_page(request):
    return render(request, 'api/brightness.html')


def create_poly_ellipsoid(request):
    folder_path = "api/stl-files/0_skin.stl"
    reader = vtk.vtkSTLReader()
    print(folder_path)
    reader.SetFileName(folder_path)
    reader.Update()

    # get the bounds of the skin mesh
    skin_poly_data = reader.GetOutput()
    bounds = skin_poly_data.GetBounds()
    print(f"Skin mesh bounds: {bounds}")

    # compute the ellipsoid radii from the mesh bounds
    x_radius = (bounds[1] - bounds[0]) / 2.0
    y_radius = (bounds[3] - bounds[2]) / 2.0
    z_radius = (bounds[5] - bounds[4]) / 2.0

    # increase the radii by 5%
    scale_factor = 1.05
    x_radius *= scale_factor 
    y_radius *= scale_factor + 0.10
    z_radius *= scale_factor

    print(f"Ellipsoid radii (increased by 5%): X={x_radius}, Y={y_radius}, Z={z_radius}")

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    skin_mapper = vtk.vtkPolyDataMapper()
    skin_mapper.SetInputConnection(reader.GetOutputPort())

    skin_actor = vtk.vtkActor()
    skin_actor.SetMapper(skin_mapper)

    # configure skin properties for a more realistic look
    skin_property = skin_actor.GetProperty()
    skin_property.SetColor(1.0, 0.8, 0.6)          # skin color
    skin_property.SetAmbient(0.2)                  # soft ambient light
    skin_property.SetDiffuse(0.8)                  # diffuse reflection
    skin_property.SetSpecular(0.3)                 # specular reflection
    skin_property.SetSpecularPower(10)             # specular highlight

    # create the ellipsoid with the adjusted radii
    ellipsoid = vtk.vtkParametricEllipsoid()
    ellipsoid.SetXRadius(x_radius)
    ellipsoid.SetYRadius(y_radius)
    ellipsoid.SetZRadius(z_radius)

    ellipsoid_source = vtk.vtkParametricFunctionSource()
    ellipsoid_source.SetParametricFunction(ellipsoid)
    ellipsoid_source.SetUResolution(50)  # ellipsoid mesh resolution
    ellipsoid_source.SetVResolution(50)
    ellipsoid_source.Update()

    ellipsoid_mapper = vtk.vtkPolyDataMapper()
    ellipsoid_mapper.SetInputConnection(ellipsoid_source.GetOutputPort())

    ellipsoid_actor = vtk.vtkActor()
    ellipsoid_actor.SetMapper(ellipsoid_mapper)
    ellipsoid_actor.GetProperty().SetColor(0.5, 0.5, 1)  # blue ellipsoid
    ellipsoid_actor.GetProperty().SetOpacity(0.5)        # semi-transparent

    # center the ellipsoid at the same position as the skin mesh
    ellipsoid_actor.SetPosition(
        (bounds[0] + bounds[1]) / 2.0,  # center X
        (bounds[2] + bounds[3]) / 2.0,  # center Y
        (bounds[4] + bounds[5]) / 2.0   # center Z
    )

    renderer.AddActor(skin_actor)
    renderer.AddActor(ellipsoid_actor)

    renderer.ResetCamera()
    render_window.Render()

    interactor.Start()
    return HttpResponse("")

def create_poly_ellipsoid_with_movement(request):
    folder_path = "api/stl-files/0_skin.stl"
    reader = vtk.vtkSTLReader()
    reader.SetFileName(folder_path)
    reader.Update()

    # get the bounds of the skin mesh
    skin_poly_data = reader.GetOutput()
    bounds = skin_poly_data.GetBounds()
    print(f"Skin mesh bounds: {bounds}")

    # compute the ellipsoid radii from the mesh bounds
    x_radius = (bounds[1] - bounds[0]) / 2.0
    y_radius = (bounds[3] - bounds[2]) / 2.0
    z_radius = (bounds[5] - bounds[4]) / 2.0

    # increase the radii by 5%
    scale_factor = 1.05
    x_radius *= scale_factor
    y_radius *= scale_factor + 0.10
    z_radius *= scale_factor

    print(f"Ellipsoid radii (increased by 5%): X={x_radius}, Y={y_radius}, Z={z_radius}")

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    skin_mapper = vtk.vtkPolyDataMapper()
    skin_mapper.SetInputConnection(reader.GetOutputPort())

    skin_actor = vtk.vtkActor()
    skin_actor.SetMapper(skin_mapper)
    skin_actor.GetProperty().SetColor(1, 0.5, 0.5)  # pink skin
    renderer.AddActor(skin_actor)

    # create the ellipsoid with the adjusted radii
    ellipsoid = vtk.vtkParametricEllipsoid()
    ellipsoid.SetXRadius(x_radius)
    ellipsoid.SetYRadius(y_radius)
    ellipsoid.SetZRadius(z_radius)

    ellipsoid_source = vtk.vtkParametricFunctionSource()
    ellipsoid_source.SetParametricFunction(ellipsoid)
    ellipsoid_source.SetUResolution(50)  # ellipsoid mesh resolution
    ellipsoid_source.SetVResolution(50)
    ellipsoid_source.Update()

    ellipsoid_mapper = vtk.vtkPolyDataMapper()
    ellipsoid_mapper.SetInputConnection(ellipsoid_source.GetOutputPort())

    ellipsoid_actor = vtk.vtkActor()
    ellipsoid_actor.SetMapper(ellipsoid_mapper)
    ellipsoid_actor.GetProperty().SetColor(0.5, 0.5, 1)  # blue ellipsoid
    ellipsoid_actor.GetProperty().SetOpacity(0.5)  # semi-transparent

    # center the ellipsoid at the same position as the skin mesh
    center_x = (bounds[0] + bounds[1]) / 2.0
    center_y = (bounds[2] + bounds[3]) / 2.0
    center_z = (bounds[4] + bounds[5]) / 2.0
    ellipsoid_actor.SetPosition(center_x, center_y, center_z)
    renderer.AddActor(ellipsoid_actor)

    # load the extra .stl that moves around the ellipsoid
    moving_stl_path = "api/stl-files/transductor y fov.stl"
    moving_reader = vtk.vtkSTLReader()
    moving_reader.SetFileName(moving_stl_path)
    moving_reader.Update()

    moving_mapper = vtk.vtkPolyDataMapper()
    moving_mapper.SetInputConnection(moving_reader.GetOutputPort())

    moving_actor = vtk.vtkActor()
    moving_actor.SetMapper(moving_mapper)
    moving_actor.GetProperty().SetColor(0, 1, 0)  # green moving object
    renderer.AddActor(moving_actor)

    # movement control variables
    theta = 0  # angle in the XY plane (radians)
    phi = 0    # angle in the XZ plane (radians)
    psi = 0    # angle in the YZ plane (radians)
    delta_angle = 0.02  # angle increment per key event
    delta_angle_psi = 0.05  # psi angle increment

    # update the position of the moving object
    def update_position():
        nonlocal theta, phi, psi
        # position on the ellipsoid surface
        x = center_x + x_radius * math.cos(theta) * math.cos(phi)
        y = center_y + y_radius * math.sin(theta) * math.cos(psi)
        z = center_z + z_radius * math.sin(phi) * math.cos(psi)
        moving_actor.SetPosition(x, y, z)
        print(f"Moving object position: x={x}, y={y}, z={z}")
        render_window.Render()

    # handle the object movement
    def move_object(obj, event):
        nonlocal theta, phi, psi
        key = obj.GetKeySym()
        if key == "Left":
            theta -= delta_angle  # clockwise in the XY plane
        elif key == "Right":
            theta += delta_angle  # counterclockwise in the XY plane
        elif key == "Up":
            phi += delta_angle  # up in the XZ plane
        elif key == "Down":
            phi -= delta_angle  # down in the XZ plane
        elif key == "a":  # 'A' moves in the YZ plane (clockwise)
            psi -= delta_angle_psi
        elif key == "d":  # 'D' moves in the YZ plane (counterclockwise)
            psi += delta_angle_psi

        # keep the angles in the range [0, 2*pi]
        theta = theta % (2 * math.pi)
        phi = phi % (2 * math.pi)
        psi = psi % (2 * math.pi)

        update_position()

    interactor.AddObserver("KeyPressEvent", move_object)

    # place the moving object at its initial position
    update_position()

    renderer.ResetCamera()
    render_window.Render()

    interactor.Start()
    return HttpResponse("")


def create_poly_spline(request):
    folder_path = "api/stl-files/skin.stl"
    reader = vtk.vtkSTLReader()
    print(folder_path)
    reader.SetFileName(folder_path)
    reader.Update()
    print(os.path.exists(reader.GetFileName()))

    # get the bounds of the skin mesh
    skin_poly_data = reader.GetOutput()
    bounds = skin_poly_data.GetBounds()
    print(f"Skin mesh bounds: {bounds}")

    # extract key points from the skin mesh
    points = vtk.vtkPoints()
    num_points = skin_poly_data.GetNumberOfPoints()
    for i in range(0, num_points, 100):  # take every 100th point
        point = skin_poly_data.GetPoint(i)
        points.InsertNextPoint(point)

    # build the spline from the key points
    spline = vtk.vtkParametricSpline()
    spline.SetPoints(points)

    # parametric function source for the spline
    spline_source = vtk.vtkParametricFunctionSource()
    spline_source.SetParametricFunction(spline)
    spline_source.SetUResolution(100)  # spline resolution
    spline_source.SetVResolution(100)
    spline_source.Update()

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # map and create an actor for the skin mesh
    skin_mapper = vtk.vtkPolyDataMapper()
    skin_mapper.SetInputConnection(reader.GetOutputPort())

    skin_actor = vtk.vtkActor()
    skin_actor.SetMapper(skin_mapper)
    skin_actor.GetProperty().SetColor(1, 0.5, 0.5)  # pink skin

    # map and create an actor for the spline
    spline_mapper = vtk.vtkPolyDataMapper()
    spline_mapper.SetInputConnection(spline_source.GetOutputPort())

    spline_actor = vtk.vtkActor()
    spline_actor.SetMapper(spline_mapper)
    spline_actor.GetProperty().SetColor(0.5, 0.5, 1)  # blue spline
    spline_actor.GetProperty().SetOpacity(0.5)  # semi-transparent

    renderer.AddActor(skin_actor)
    renderer.AddActor(spline_actor)

    renderer.ResetCamera()
    render_window.Render()

    interactor.Start()
    return HttpResponse("")

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

mallas = []

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
    # Encuentra el resolver que corresponde a la aplicación "api"
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
    return JsonResponse(files, safe=False)

###VTK visulization para ver el recorte

def levantarMallas():
    # Load a mesh from a file
    folder_path = "api/stl-files"
    stl_files = [f for f in os.listdir(folder_path) if f.endswith('.stl')]

    if not stl_files:
        print("No STL files found in the directory.")
        return
    for stl_file in stl_files:
        reader = vtk.vtkSTLReader()
        reader.SetFileName(os.path.join(folder_path, stl_file))
        reader.Update()
        mallas.append(reader.GetOutput())
    return

def update_visualization(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        normal = data.get('normal', [0.3, 0.5, 1])
        vtk_visualization(request,normal)
        return JsonResponse({'success': True})
    return JsonResponse({'success': False})


def vtk_visualization(request, normal=[0.3, 0.5, 1]):
    

    # Set up VTK rendering
    renderer = vtk.vtkRenderer()
    if (not mallas):
        levantarMallas()
    for malla in mallas:

        liver_mesh = malla

        # Get the filled slice for each mesh
        filled_slice = slice_and_fill_mesh_vtk(liver_mesh, origin=[0, 0, -0.02], normal=normal)

        # Mapper and actor for each STL mesh
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(liver_mesh)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.1)

        # Mapper and actor for the filled slice
        slice_mapper = vtk.vtkPolyDataMapper()
        slice_mapper.SetInputData(filled_slice)

        slice_actor = vtk.vtkActor()
        slice_actor.SetMapper(slice_mapper)
        slice_actor.GetProperty().SetColor(0.58, 0.0, 0.83)

        # Add the actors to the scene
        renderer.AddActor(actor)
        renderer.AddActor(slice_actor)
    
    renderer.SetBackground(1, 1, 1)

    # Create a render window and interactor
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # Set the size of the render window
    render_window.SetSize(1200, 800)  # Width and height in pixels

    # Create a render window interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Render and interact
    render_window.Render()
    
    # Attempt to bring the window to the front
    render_window.SetWindowName("VTK Visualization")
    render_window.Render()
    render_window_interactor.Initialize()
    render_window_interactor.Start()

    return True
    # return render(request, 'api/vtk_visualization.html')


####SLICE AND FILL




###########fovmesh

def create_fov_mesh(origin, normal, radius, height, angle):
    # Normalize the normal vector
    norm = np.linalg.norm(normal)
    normal = [normal[0] / norm, normal[1] / norm, normal[2] / norm]
    
    # Create rotation matrix to align FOV with the given normal
    up = [1, 1, 0]  # Default up direction
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
    
# Create the FOV polygon
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
    
    # Create a polydata to mesh filter to generate the FOV mesh
    fov_mesh = vtk.vtkPolyData()
    fov_mesh.DeepCopy(fov_polydata)
    
    return fov_mesh


def mallaDelFOV(request):
    # Load a mesh from a file
    folder_path = "api/stl-files"
    stl_files = [f for f in os.listdir(folder_path) if f.endswith('.stl')]

    if not stl_files:
        print("No STL files found in the directory.")
        return

    # Set up VTK rendering
    renderer = vtk.vtkRenderer()

    for stl_file in stl_files:
        reader = vtk.vtkSTLReader()
        reader.SetFileName(os.path.join(folder_path, stl_file))
        reader.Update()
        liver_mesh = reader.GetOutput()

        # Get the filled slice for each mesh
        #filled_slice = slice_and_fill_mesh_vtk(liver_mesh, origin=[0, 0, -0.02], normal=normal)

        # Mapper and actor for each STL mesh
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(liver_mesh)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.1)

        # Mapper and actor for the filled slice
        #slice_mapper = vtk.vtkPolyDataMapper()
        #slice_mapper.SetInputData(filled_slice)

        #slice_actor = vtk.vtkActor()
        #slice_actor.SetMapper(slice_mapper)
        #slice_actor.GetProperty().SetColor(0.58, 0.0, 0.83)

        # Add the actors to the scene
        renderer.AddActor(actor)
        #renderer.AddActor(slice_actor)
    
    


    ## Define the origin and normal for the FOV
    slice_origin = [0.,0.,-0.02]
    slice_normal=[-5.,-20.,1.]

    # Define FOV shape parameters
    radius = 1
    height = 0.1
    angle = math.radians(60)  # Convert degrees to radians

    # Create FOV mask on the same plane as slice origin and normal (No esta en el mismo plano que el slice- a corregir)
    fov_mesh = create_fov_mesh(slice_origin, slice_normal, radius, height, angle)

    # Mapper and actor for the FOV mesh
    fov_mapper = vtk.vtkPolyDataMapper()
    fov_mapper.SetInputData(fov_mesh)
    fov_actor = vtk.vtkActor()
    fov_actor.SetMapper(fov_mapper)
    fov_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Set color to red
    fov_actor.GetProperty().SetOpacity(1)
    


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






#### simeco frontend
def slice_and_fill_mesh_vtk(mesh, origin=(0, 0, 0), normal=(1, 0, 0)):
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




def slice_to_image(filled_slice):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(filled_slice)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(100/255.0, 0.0, 100/255.0)  
    actor.GetProperty().LightingOff()  

    # Create a renderer and render window
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # Add the actor to the scene
    renderer.AddActor(actor)
    renderer.SetBackground(0, 0, 0)  # Set background to black

    # Create a render window interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Render the scene
    render_window.Render()

    # Capture the image
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.Update()

    vtk_image = window_to_image_filter.GetOutput()

    # Convert VTK image to numpy array
    width, height, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    components = vtk_array.GetNumberOfComponents()
    arr = vtk_to_numpy(vtk_array).reshape(height, width, components)

    # Save image using PIL
    output_path = "./results/imagenesTomadasDeVTK/"
    os.makedirs(output_path, exist_ok=True)
    image = Image.fromarray(arr.astype('uint8'), 'RGB')
    image.save(os.path.join(output_path, 'views_save_image.png'))

    return arr




def vtk_visualization_image(request): #de /front es el que hace todo
    #Load all meshes from the folder
    data = json.loads(request)
    x = data['x']
    y = data['y']
    z = data['z']

    
    append_filter = vtk.vtkAppendPolyData()
    if (not mallas):
        levantarMallas()
    for malla in mallas:
        append_filter.AddInputData(malla)

    append_filter.Update()
    combined_mesh = append_filter.GetOutput()

    # Perform the slicing on the combined mesh
    filled_slice = slice_and_fill_mesh_vtk(combined_mesh, origin=[0, 0, -0.02], normal=[x, y, z])

    # Convert filled slice to image
    slice_image = slice_to_image(filled_slice)
    
    
    # Return HTML response
    return slice_image


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
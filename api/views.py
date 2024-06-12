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


def update_visualization(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        normal = data.get('normal', [0.3, 0.5, 1])
        vtk_visualization(request,normal)
        return JsonResponse({'success': True})
    return JsonResponse({'success': False})

def vtk_visualization(request,normal=[0.3, 0.5, 1]):
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
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Render and interact
    render_window.Render()
    render_window_interactor.Start()

    # Return rendered template
    return render(request, 'api/vtk_visualization.html')


####SLICE AND FILL


""" ###########VER SI ESTO SE PEUDE BORRAR
def save_image(request):
    if request.method == 'POST':
        try:
            data = request.json()
            image_data = data.get('image', '')
            if not image_data:
                return JsonResponse({'error': 'No image data found'}, status=400)

            # Decoding the base64 image data
            format, imgstr = image_data.split(';base64,')
            ext = format.split('/')[-1]
            img_data = base64.b64decode(imgstr)

            # Define the path to save the image
            image_path = os.path.join(settings.MEDIA_ROOT, 'slices', 'slice.png')

            # Ensure the directory exists
            os.makedirs(os.path.dirname(image_path), exist_ok=True)

            # Write the image data to the file
            with open(image_path, 'wb') as f:
                f.write(img_data)

            return JsonResponse({'message': 'Image saved successfully'})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=405)

"""



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

    folder_path = "api/stl-files"
    stl_files = [f for f in os.listdir(folder_path) if f.endswith('.stl')]
    
    if not stl_files:
        print("No STL files found in the directory.")
        return HttpResponse("No STL files found in the directory.")

    # Create a vtkAppendPolyData to combine all meshes
    append_filter = vtk.vtkAppendPolyData()

    for stl_file in stl_files:
        reader = vtk.vtkSTLReader()
        reader.SetFileName(os.path.join(folder_path, stl_file))
        reader.Update()
        append_filter.AddInputData(reader.GetOutput())

    append_filter.Update()
    combined_mesh = append_filter.GetOutput()

    # Perform the slicing on the combined mesh
    filled_slice = slice_and_fill_mesh_vtk(combined_mesh, origin=[0, 0, -0.02], normal=[x, y, z])

    # Convert filled slice to image
    slice_image = slice_to_image(filled_slice)
    
    
    # Return HTML response
    return slice_image



def vtk_image(request):
    return render(request, 'api/vtk_image.html')

def pruebaFOV(request):
    return render(request, 'api/fov.html')

def red128(request):
    return render(request, 'api/red128.html')

def red256(request):
    return render(request, 'api/red256.html')


import vtk
import os

# Actores globales
transducer_actor = None
mallas_actores = []

# Renderizado
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# Colores por nombre
mesh_colors = {
    "higado": (0.8, 0.4, 0.4),
    "estomago": (0.4, 0.8, 0.4),
    "intestino": (0.4, 0.4, 0.8),
    # etc...
}

def levantarMallas():
    global transducer_actor, mallas_actores

    folder_path = "api/stl-files"
    stl_files = [f for f in os.listdir(folder_path) if f.endswith('.stl')]

    for stl_file in stl_files:
        reader = vtk.vtkSTLReader()
        reader.SetFileName(os.path.join(folder_path, stl_file))
        reader.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        name = os.path.splitext(stl_file)[0].lower()
        assigned_color = (1.0, 1.0, 1.0)
        for keyword, color in mesh_colors.items():
            if keyword in name:
                assigned_color = color
                break
        actor.GetProperty().SetColor(assigned_color)

        if "transductor" in name:
            transducer_actor = actor
        else:
            mallas_actores.append(actor)

    if transducer_actor:
        renderer.AddActor(transducer_actor)
    for actor in mallas_actores:
        renderer.AddActor(actor)

    renderer.SetBackground(0.1, 0.1, 0.1)
    render_window.Render()
    interactor.Initialize()
    interactor.Start()


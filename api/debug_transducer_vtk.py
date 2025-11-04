import vtk
import math
import numpy as np


def euler_to_rotation_matrix(orientation):
    """Convierte (yaw, pitch, roll) en matriz de rotación 3x3 (en radianes)."""
    yaw, pitch, roll = np.radians(orientation)

    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw),  math.cos(yaw), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll),  math.cos(roll)]
    ])

    return Rz @ Ry @ Rx


def make_arrow(start, direction, color=(1, 0, 0), scale=1.0):
    """Crea una flecha desde start apuntando en direction."""
    norm = np.linalg.norm(direction)
    if norm == 0:
        return None

    direction = np.array(direction) / norm
    arrow_source = vtk.vtkArrowSource()
    arrow_source.SetTipLength(0.3)
    arrow_source.SetShaftRadius(0.01)

    transform = vtk.vtkTransform()
    default_dir = np.array([1, 0, 0])
    axis = np.cross(default_dir, direction)
    angle = math.degrees(math.acos(np.clip(np.dot(default_dir, direction), -1.0, 1.0)))
    if np.linalg.norm(axis) > 0.0:
        transform.RotateWXYZ(angle, *axis)
    transform.Scale(norm * scale, norm * scale, norm * scale)
    transform.Translate(start)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputConnection(arrow_source.GetOutputPort())
    transform_filter.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(transform_filter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    return actor


def make_cone(start, direction, color=(0.8, 0.2, 0.2), scale=1.0):
    """Crea un cono que representa el transductor, apuntando en direction."""
    norm = np.linalg.norm(direction)
    if norm == 0:
        return None

    direction = np.array(direction) / norm
    cone_source = vtk.vtkConeSource()
    cone_source.SetResolution(20)
    cone_source.SetHeight(1.0 * scale)
    cone_source.SetRadius(0.3 * scale)
    cone_source.SetDirection(direction)  # ya apunta correctamente
    cone_source.SetCenter(start + direction * 0.5 * scale)  # centro del cono
    cone_source.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(cone_source.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    return actor


def debug_transducer_vtk(
    position,
    orientation,
    normal,
    target_origin,
    show_skin=True,
    show_position=True,
    show_normal=True,
    show_plane=True,
    show_target=True,
    show_orientation=True,
    show_cone=True,
    scale_factor=1.0
):
    """Visualiza el transductor y elementos geométricos."""
    
    print("=== LEYENDA DEBUG VTK ===")
    print("🟡 ESFERA AMARILLA: Posición del TRANSDUCTOR")
    print("🟢 ESFERA VERDE: Origen del PLANO DE CORTE") 
    print("⚪ FLECHA BLANCA: Normal del plano (dirección del plano)")
    print("🔴 FLECHA ROJA: Eje X local del transductor")
    print("🟢 FLECHA VERDE: Eje Y local del transductor") 
    print("🔵 FLECHA AZUL: Eje Z local del transductor")
    print("🔺 CONO ROJO: Dirección en la que 'mira' el transductor")
    print("📐 PLANO GRIS: Plano de corte que se usa para el recorte VTK")
    print("==========================")

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # === Piel ===
    if show_skin:
        skin_reader = vtk.vtkSTLReader()
        skin_reader.SetFileName("api/stl_para usar/0_skin.stl")
        skin_reader.Update()

        skin_mapper = vtk.vtkPolyDataMapper()
        skin_mapper.SetInputConnection(skin_reader.GetOutputPort())

        skin_actor = vtk.vtkActor()
        skin_actor.SetMapper(skin_mapper)
        skin_actor.GetProperty().SetOpacity(0.15)
        skin_actor.GetProperty().SetColor(1, 0.7, 0.6)
        renderer.AddActor(skin_actor)

    # === Posición del transductor (AMARILLO) ===
    if show_position:
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(*position)
        sphere.SetRadius(3.0 * scale_factor)
        sphere.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 1, 0)  # AMARILLO - TRANSDUCTOR
        renderer.AddActor(actor)

    # === Flecha de la normal (BLANCA) ===
    if show_normal:
        # La normal es la dirección del PLANO (no del transductor)
        normal_actor = make_arrow(position, np.array(normal) * 30 * scale_factor, color=(1, 1, 1))  # BLANCO - NORMAL DEL PLANO
        if normal_actor:
            renderer.AddActor(normal_actor)

    # === Plano de corte (GRIS) ===
    if show_plane:
        origin = target_origin  # target_origin es el origin del plano en este contexto
        plane = vtk.vtkPlaneSource()
        plane.SetCenter(*origin)
        plane.SetNormal(*normal)
        plane.SetXResolution(10)
        plane.SetYResolution(10)
        s = 50 * scale_factor
        plane.SetOrigin(origin[0] - s, origin[1] - s, origin[2])
        plane.SetPoint1(origin[0] + s, origin[1] - s, origin[2])
        plane.SetPoint2(origin[0] - s, origin[1] + s, origin[2])
        plane.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.7, 0.7, 0.7)  # GRIS - PLANO DE CORTE
        actor.GetProperty().SetOpacity(0.3)
        renderer.AddActor(actor)

    # === Target origin (VERDE) ===
    if show_target:
        target = vtk.vtkSphereSource()
        target.SetCenter(*target_origin)
        target.SetRadius(2.0 * scale_factor)
        target.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(target.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0, 1, 0)  # VERDE - ORIGEN DEL PLANO
        renderer.AddActor(actor)

    # === Ejes locales ===
    if show_orientation:
    # 'orientation' param puede ser una tupla de eulers o una matriz aplanada (9 elems)
        if isinstance(orientation, (list, tuple)) and len(orientation) == 9:
            R = np.array(orientation).reshape((3,3))
        else:
            # si orientation es euler (3 elems), convertimos con la función existente:
            R = euler_to_rotation_matrix(orientation)

        x_axis, y_axis, z_axis = R[:, 0], R[:, 1], R[:, 2]

        # ROJO: Eje X
        actor_x = make_arrow(position, x_axis * 20 * scale_factor, color=(1, 0, 0))
        # VERDE: Eje Y  
        actor_y = make_arrow(position, y_axis * 20 * scale_factor, color=(0, 1, 0))
        # AZUL: Eje Z
        actor_z = make_arrow(position, z_axis * 20 * scale_factor, color=(0, 0, 1))
        
        if actor_x: renderer.AddActor(actor_x)
        if actor_y: renderer.AddActor(actor_y) 
        if actor_z: renderer.AddActor(actor_z)

    # === Cono del transductor (ROJO) - DEBE APUNTAR HACIA EL PLANO ===
    if show_cone:
        # El cono debe apuntar en la dirección OPUESTA a la normal del plano
        # porque la normal apunta HACIA el transductor, pero el cono apunta HACIA EL PLANO
        direction = -np.array(normal)  # Dirección hacia el plano
        cone_actor = make_cone(np.array(position), direction, color=(0.8, 0.2, 0.2), scale=10*scale_factor)
        if cone_actor:
            renderer.AddActor(cone_actor)

    # === Render final ===
    renderer.SetBackground(0.1, 0.1, 0.15)
    render_window.SetSize(800, 600)
    renderer.ResetCamera()
    render_window.Render()
    interactor.Start()


# === Ejemplo de uso ===
if __name__ == "__main__":
    debug_transducer_vtk(
        position=(-0.25, 0.0, 0.0),
        orientation=(0.0, 90.0, 0.0),
        normal=(0.0, -1.0, 0.0),
        target_origin=(-0.25, -0.25, 0.0),
        show_skin=True,
        show_position=True,
        show_normal=True,
        show_plane=True,
        show_target=True,
        show_orientation=True,
        show_cone=True,
        scale_factor=0.02  # Ajusta para ver mejor el tamaño de flechas y cono
    )

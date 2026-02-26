import vtk
import numpy as np
from vtk.util import numpy_support

class CoronalSliceVisualizer:
    """
    Visualizador de cortes coronales a partir de mallas VTK y un controlador
    que proporciona posición y orientación (pitch, yaw, roll) en radianes.

    El plano coronal se define con origen = posición del controlador y normal = up,
    donde up se obtiene de la base ortonormal construida desde la rotación.
    """

    def __init__(self, meshes, width, height):
        """
        Parámetros
        ----------
        meshes : list of (vtkPolyData, tuple)
            Lista de tuplas (malla, color) donde color es (r,g,b) en [0,1].
        width : int
            Ancho de la imagen de salida en píxeles.
        height : int
            Alto de la imagen de salida en píxeles.
        """
        self.meshes = meshes
        self.width = width
        self.height = height

        # Configuración del renderizador off‑screen
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0, 0, 0)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetOffScreenRendering(1)
        self.render_window.SetSize(width, height)

        # Crear actores (uno por malla) con mappers vacíos inicialmente
        self.actors = []  # cada elemento: (actor, malla_original)
        for mesh, color in meshes:
            mapper = vtk.vtkPolyDataMapper()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color[0], color[1], color[2])
            self.renderer.AddActor(actor)
            self.actors.append((actor, mesh))

        # Configurar cámara en modo proyección paralela
        self.camera = self.renderer.GetActiveCamera()
        self.camera.SetParallelProjection(True)

    def _euler_to_matrix(self, pitch, yaw, roll):
        """
        Construye la matriz de rotación 3x3 (right‑handed) con el orden:
        Rz(roll) @ Ry(yaw) @ Rx(pitch)

        Devuelve una matriz numpy de 3x3.
        """
        # Rotación alrededor de X (pitch)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch),  np.cos(pitch)]])
        # Rotación alrededor de Y (yaw)
        Ry = np.array([[ np.cos(yaw), 0, np.sin(yaw)],
                       [0, 1, 0],
                       [-np.sin(yaw), 0, np.cos(yaw)]])
        # Rotación alrededor de Z (roll)
        Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                       [np.sin(roll),  np.cos(roll), 0],
                       [0, 0, 1]])
        return Rz @ Ry @ Rx

    def slice_and_fill_mesh_vtk(self, mesh, origin, normal):
        """
        Corta una malla con un plano y rellena la intersección para obtener
        una superficie (polígono relleno). Si no hay intersección devuelve
        un vtkPolyData vacío.

        Parámetros
        ----------
        mesh : vtkPolyData
            Malla de entrada.
        origin : array_like (3,)
            Punto de origen del plano.
        normal : array_like (3,)
            Vector normal del plano.

        Devuelve
        --------
        vtkPolyData
            Geometría de la intersección rellena (triángulos) o vacía.
        """
        # Plano de corte
        plane = vtk.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)

        # Generar las líneas de intersección
        cutter = vtk.vtkCutter()
        cutter.SetInputData(mesh)
        cutter.SetCutFunction(plane)
        cutter.Update()

        contour_lines = cutter.GetOutput()
        if contour_lines.GetNumberOfCells() == 0:
            # Sin intersección
            empty = vtk.vtkPolyData()
            return empty

        # Rellenar los contornos cerrados para obtener una superficie
        triangulator = vtk.vtkContourTriangulator()
        triangulator.SetInputData(contour_lines)
        triangulator.Update()
        return triangulator.GetOutput()

    def render_from_controller(self, controller_mov):
        """
        Genera una imagen del corte coronal según la posición y orientación
        actuales del controlador.

        Parámetros
        ----------
        controller_mov : objeto con métodos:
            get_current_position() -> (x, y, z)
            get_current_rotation() -> (pitch, yaw, roll)  # radianes

        Devuelve
        --------
        image_numpy_uint8 : numpy.ndarray
            Imagen RGB de tamaño (height, width, 3) con valores uint8.
        position : tuple
            (x, y, z) usados para el corte.
        rotation : tuple
            (pitch, yaw, roll) usados para el corte.
        """
        # Obtener datos del controlador
        
        position = controller_mov.get_current_position()
        pitch, yaw, roll = controller_mov.get_current_orientation()
        print("Obteniendo posición ", position)
        print("Obteniendo orientación (pitch, yaw, roll) en radianes: ", (pitch, yaw, roll))
        # Construir base ortonormal
        # Dentro de render_from_controller, antes de llamar a _euler_to_matrix:
        pitch = np.radians(pitch)   # convierte grados → radianes
        yaw   = np.radians(yaw)
        roll  = np.radians(roll)
        R = self._euler_to_matrix(pitch, yaw, roll)

        forward_local = np.array([0, 0, 1])
        right_local   = np.array([1, 0, 0])
        up_local      = np.array([0, 1, 0])

        forward = R @ forward_local
        right   = R @ right_local
        up      = R @ up_local

        # Normalizar (por si acaso, aunque R sea de rotación pura)
        forward = forward / np.linalg.norm(forward)
        right   = right / np.linalg.norm(right)
        up      = up / np.linalg.norm(up)

        offset = -1
        origin = np.array(position) + forward * offset
        normal = up  # normal del plano coronal

        # Lista para bounds de geometrías visibles
        visible_bounds = []
        visible_count = 0

        # Aplicar slice a cada malla y actualizar actores
        for actor, mesh in self.actors:
            sliced = self.slice_and_fill_mesh_vtk(mesh, origin, normal)
            if sliced.GetNumberOfCells() == 0:
                actor.VisibilityOff()
            else:
                actor.VisibilityOn()
                actor.GetMapper().SetInputData(sliced)
                bounds = sliced.GetBounds()
                # Verificar que el bounds sea válido (xmin <= xmax)
                if bounds[1] >= bounds[0]:
                    visible_bounds.append(bounds)
                    visible_count += 1

        # Si no hay nada visible, devolver imagen negra
        if visible_count == 0:
            black_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return black_img, position, (pitch, yaw, roll)

        # Combinar bounds globales
        global_bounds = list(visible_bounds[0])
        for bounds in visible_bounds[1:]:
            for i in range(6):
                if i % 2 == 0:  # mínimo
                    global_bounds[i] = min(global_bounds[i], bounds[i])
                else:            # máximo
                    global_bounds[i] = max(global_bounds[i], bounds[i])

        # Diagonal del bounding box para calcular distancia de cámara
        dx = global_bounds[1] - global_bounds[0]
        dy = global_bounds[3] - global_bounds[2]
        dz = global_bounds[5] - global_bounds[4]
        diagonal = np.sqrt(dx*dx + dy*dy + dz*dz)
        distancia = 2.0 * diagonal

        # Calcular extensiones en el plano (direcciones right y forward)
        # Para ello transformamos los 8 vértices del bounding box
        # al sistema local con ejes (right, up, forward)
        corners = []
        for x in (global_bounds[0], global_bounds[1]):
            for y in (global_bounds[2], global_bounds[3]):
                for z in (global_bounds[4], global_bounds[5]):
                    corners.append(np.array([x, y, z]))

        # Matriz inversa (transpuesta) para pasar de mundo a local
        R_inv = R.T
        local_corners = [R_inv @ (c - origin) for c in corners]

        # Componentes right (x local) y forward (z local)
        right_vals = [c[0] for c in local_corners]
        forward_vals = [c[2] for c in local_corners]
        width_bb = max(right_vals) - min(right_vals)
        height_bb = max(forward_vals) - min(forward_vals)

        # Configurar cámara
        self.camera.SetPosition(origin - up * distancia)
        self.camera.SetFocalPoint(origin)
        self.camera.SetViewUp(forward)
        self.camera.SetParallelScale(max(width_bb, height_bb) * 0.6)

        # Renderizar
        self.render_window.Render()

        # Capturar imagen
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(self.render_window)
        w2if.Update()
        vtk_image = w2if.GetOutput()

        # Convertir a numpy array (asumimos 3 canales RGB)
        vtk_array = vtk_image.GetPointData().GetScalars()
        np_image = numpy_support.vtk_to_numpy(vtk_array)
        dims = vtk_image.GetDimensions()  # (width, height, 1)
        np_image = np_image.reshape(dims[1], dims[0], 3)
        np_image = np_image.astype(np.uint8)

        np_image = np.fliplr(np_image)
        return np_image, position, (pitch, yaw, roll)
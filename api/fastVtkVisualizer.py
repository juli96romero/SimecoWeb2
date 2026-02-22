# api/fastVtkVisualizer.py
import math
import numpy as np
import vtk
from vtk.util import numpy_support

class FastVtkVisualizer:
    """
    Reimplementación persistente del pipeline VTK que mantiene el
    comportamiento EXACTO de la función original vtk_visualization_images.
    - Conserva la lógica de rotaciones y framing (cálculo de bounds por frame).
    - Evita recrear actores/mappers/clipper/renderer cada frame; los crea una vez.
    - Sigue haciendo clipper.Update() y append_all.Update() por frame para que
      la cámara quede idéntica al original.
    """

    def __init__(self, mallas, width=300, height=300):
        """
        mallas: lista de (vtkPolyData, color) -- mismo formato que tenías
        width, height: tamaño de la imagen de salida
        """
        self.mallas = mallas
        self.width = width
        self.height = height

        # Renderer + RenderWindow (creados una sola vez)
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(0, 0, 0)

        self.renWin = vtk.vtkRenderWindow()
        self.renWin.SetOffScreenRendering(1)
        self.renWin.AddRenderer(self.ren)
        self.renWin.SetSize(self.width, self.height)
        # Desactivar multi sampling por perf (igual que en original no estaba explícito)
        self.renWin.SetMultiSamples(0)

        # Plano persistente
        self.plane = vtk.vtkPlane()

        # Para cada malla: creamos un clipper conectado al polydata original,
        # un mapper que tome la salida del clipper y un actor. No vamos a
        # recrearlos por frame: sólo actualizamos la función de clip (plane)
        self.clippers = []
        self.mappers = []
        self.actors = []

        for polydata, color in self.mallas:
            clipper = vtk.vtkClipPolyData()
            clipper.SetInputData(polydata)
            clipper.SetClipFunction(self.plane)
            # NOTA: no llamamos clipper.Update() aquí (se hará por frame)

            mapper = vtk.vtkPolyDataMapper()
            # Conectamos al puerto de salida del clipper para que el pipeline sea dinámico
            mapper.SetInputConnection(clipper.GetOutputPort())
            mapper.ScalarVisibilityOff()

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color)
            actor.GetProperty().SetLighting(False)

            self.ren.AddActor(actor)

            self.clippers.append(clipper)
            self.mappers.append(mapper)
            self.actors.append(actor)

        # Cámara (paralela como en original)
        self.camera = self.ren.GetActiveCamera()
        self.camera.SetParallelProjection(True)

        # WindowToImageFilter persistente
        self.w2if = vtk.vtkWindowToImageFilter()
        self.w2if.SetInput(self.renWin)
        self.w2if.SetInputBufferTypeToRGB()
        self.w2if.ReadFrontBufferOff()

    def vtk_visualization_images(self, mov_module, image_rotation_deg=0):
        """
        Firma idéntica a tu función original:
            img, pos, rot = vtk_engine.vtk_visualization_images(mov, image_rotation_deg=90)
        Devuelve: (img_uint8, pos, rot)
        """

        # IMPORTS LOCALES (igual que original)
        import math as _math
        import numpy as _np

        # -------------------------------------------------
        # Obtener controlador (igual que original)
        # -------------------------------------------------
        controller = mov_module._controller
        if controller is None:
            mov_module.init_controller()
            controller = mov_module._controller

        pos = controller.calculate_position()
        rot = controller.calculate_orientation()

        pos_np = _np.array(pos)

        # -------------------------------------------------
        # Calcular forward/up exactamente como tu original
        # -------------------------------------------------
        center_proj = _np.array([
            controller.center_x,
            pos_np[1],  # ← misma altura actual
            controller.center_z
        ])

        forward = center_proj - pos_np
        # defensiva: si por alguna razón forward es cero, evitar crash
        norm_forward = _np.linalg.norm(forward)
        if norm_forward < 1e-9:
            forward = _np.array([0.0, 0.0, 1.0])
        else:
            forward = forward / norm_forward

        world_up = _np.array([0, 1, 0])
        if abs(_np.dot(forward, world_up)) > 0.99:
            world_up = _np.array([0, 0, 1])

        right = _np.cross(forward, world_up)
        right = right / _np.linalg.norm(right)

        up = _np.cross(right, forward)
        up = up / _np.linalg.norm(up)

        # -------------------------------------------------
        # Rotaciones locales (misma matemática que original)
        # -------------------------------------------------
        pitch = controller.local_pitch
        yaw   = controller.local_yaw
        roll  = controller.local_roll

        def rot_x(a):
            return _np.array([
                [1, 0, 0],
                [0, _np.cos(a), -_np.sin(a)],
                [0, _np.sin(a),  _np.cos(a)]
            ])

        def rot_y(a):
            return _np.array([
                [ _np.cos(a), 0, _np.sin(a)],
                [0, 1, 0],
                [-_np.sin(a), 0, _np.cos(a)]
            ])

        def rot_z(a):
            return _np.array([
                [_np.cos(a), -_np.sin(a), 0],
                [_np.sin(a),  _np.cos(a), 0],
                [0, 0, 1]
            ])

        R = rot_z(roll) @ rot_y(yaw) @ rot_x(pitch)

        forward = R @ forward
        up = R @ up

        # -------------------------------------------------
        # Actualizar plano de corte (igual que original)
        # -------------------------------------------------
        self.plane.SetOrigin(pos)
        self.plane.SetNormal(up.tolist())

        # -------------------------------------------------
        # Para cada malla: ejecutar clipper.Update() y recolectar polígonos rellenados
        # (igual que en tu código original, que hacía clipper.Update() y
        # mapper.SetInputData(filled_poly))
        # -------------------------------------------------
        actores_creados = 0
        append_all = vtk.vtkAppendPolyData()

        for idx, (clipper, actor) in enumerate(zip(self.clippers, self.actors)):
            # El clipper ya está conectado al polydata original y a la plane
            clipper.Update()  # produce el filled_poly dinámico
            filled_poly = clipper.GetOutput()

            if filled_poly is None or filled_poly.GetNumberOfCells() == 0:
                # ocultar actor para que no participe en el render
                actor.SetVisibility(False)
                continue

            # Si tiene células, asegurarnos que el actor esté visible
            actor.SetVisibility(True)
            actores_creados += 1

            # Añadir el polydata recortado al append para calcular bounds
            append_all.AddInputData(filled_poly)

        if actores_creados == 0:
            # misma conducta que tu original: imagen negra de 300x300 y return
            img = _np.zeros((self.height, self.width, 3), dtype=_np.uint8)
            return img, pos, rot

        # -------------------------------------------------
        # Bounds combinados (igual que original)
        # -------------------------------------------------
        append_all.Update()
        combined = append_all.GetOutput()
        bounds = combined.GetBounds()

        dx_b = bounds[1] - bounds[0]
        dy_b = bounds[3] - bounds[2]
        dz_b = bounds[5] - bounds[4]

        diagonal = _math.sqrt(dx_b*dx_b + dy_b*dy_b + dz_b*dz_b)
        if diagonal < 1e-6:
            diagonal = 1.0

        # -------------------------------------------------
        # Cámara (idéntica a original)
        # -------------------------------------------------
        # Asegurarnos de usar la misma renWin que creamos en __init__
        self.renWin.SetOffScreenRendering(1)
        # tamaño (igual al original)
        self.renWin.SetSize(self.width, self.height)

        camera = self.ren.GetActiveCamera()

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

        self.ren.ResetCameraClippingRange()

        # -------------------------------------------------
        # Render (igual que original)
        # -------------------------------------------------
        self.renWin.Render()

        # -------------------------------------------------
        # Capturar imagen (igual que original)
        # -------------------------------------------------
        w2if = self.w2if  # persistente
        w2if.SetInput(self.renWin)  # redundante pero mantiene idéntico el pipeline
        w2if.SetInputBufferTypeToRGB()
        w2if.ReadFrontBufferOff()
        w2if.Modified()
        w2if.Update()

        vtk_image = w2if.GetOutput()
        dims = vtk_image.GetDimensions()

        vtk_array = vtk_image.GetPointData().GetScalars()
        np_array = numpy_support.vtk_to_numpy(vtk_array)

        img = np_array.reshape(dims[1], dims[0], 3)
        img = np.flipud(img)

        return img.astype(np.uint8), pos, rot
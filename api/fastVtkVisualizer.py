# api/fastVtkVisualizer.py
import math
import numpy as np
import vtk
from vtk.util import numpy_support
import random


class FastVtkVisualizer:
    """
    Pipeline VTK persistente optimizado.
    Mantiene comportamiento EXACTO del original.
    """

    def __init__(self, mallas, width=300, height=300):
        self.mallas = mallas
        self.width = width
        self.height = height

        # Renderer
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(0, 0, 0)

        # RenderWindow persistente
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.SetOffScreenRendering(1)
        self.renWin.SetMultiSamples(0)
        self.renWin.AddRenderer(self.ren)
        self.renWin.SetSize(self.width, self.height)

        # Plano persistente
        self.plane = vtk.vtkPlane()

        self.clippers = []
        self.actors = []

        for polydata, color in self.mallas:
            clipper = vtk.vtkClipPolyData()
            clipper.SetInputData(polydata)
            clipper.SetClipFunction(self.plane)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(clipper.GetOutputPort())
            mapper.ScalarVisibilityOff()

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color)
            actor.GetProperty().SetLighting(False)

            self.ren.AddActor(actor)

            self.clippers.append(clipper)
            self.actors.append(actor)

        # Cámara paralela
        self.camera = self.ren.GetActiveCamera()
        self.camera.SetParallelProjection(True)

        # WindowToImage persistente
        self.w2if = vtk.vtkWindowToImageFilter()
        self.w2if.SetInput(self.renWin)
        self.w2if.SetInputBufferTypeToRGB()
        self.w2if.ReadFrontBufferOff()

    # ---------------------------------------------------------
    # ROTACIONES OPTIMIZADAS (sin crear matrices 3x3)
    # ---------------------------------------------------------

    @staticmethod
    def apply_euler_rotation(forward, up, pitch, yaw, roll):
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cr, sr = math.cos(roll), math.sin(roll)

        # rot_x (pitch)
        forward = np.array([
            forward[0],
            cp * forward[1] - sp * forward[2],
            sp * forward[1] + cp * forward[2]
        ])

        up = np.array([
            up[0],
            cp * up[1] - sp * up[2],
            sp * up[1] + cp * up[2]
        ])

        # rot_y (yaw)
        forward = np.array([
            cy * forward[0] + sy * forward[2],
            forward[1],
            -sy * forward[0] + cy * forward[2]
        ])

        up = np.array([
            cy * up[0] + sy * up[2],
            up[1],
            -sy * up[0] + cy * up[2]
        ])

        # rot_z (roll)
        forward = np.array([
            cr * forward[0] - sr * forward[1],
            sr * forward[0] + cr * forward[1],
            forward[2]
        ])

        up = np.array([
            cr * up[0] - sr * up[1],
            sr * up[0] + cr * up[1],
            up[2]
        ])

        return forward, up

    # ---------------------------------------------------------
    # FUNCIÓN PRINCIPAL
    # ---------------------------------------------------------

    def vtk_visualization_images(self, mov_module, image_rotation_deg=0):

        controller = mov_module._controller
        if controller is None:
            mov_module.init_controller()
            controller = mov_module._controller

        rand_pos = (
            random.choice([-0.002, -0.001, 0, 0.001, 0.002]),
            random.choice([-0.002, -0.001, 0, 0.001, 0.002]),
            random.choice([-0.002, -0.001, 0, 0.001, 0.002])
        )

        pos = controller.calculate_position()
        rot = controller.calculate_orientation()

        pos = (
            pos[0] + rand_pos[0],
            pos[1] + rand_pos[1],
            pos[2] + rand_pos[2]
        )

        pos_np = np.asarray(pos, dtype=np.float64)

        # -------------------------------------------------
        # Forward / Up base
        # -------------------------------------------------

        center_proj = np.array([
            controller.center_x,
            pos_np[1],
            controller.center_z
        ], dtype=np.float64)

        forward = center_proj - pos_np
        norm = np.linalg.norm(forward)

        if norm < 1e-9:
            forward = np.array([0.0, 0.0, 1.0])
        else:
            forward /= norm

        world_up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([0.0, 0.0, 1.0])

        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)

        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        # -------------------------------------------------
        # Rotación Euler optimizada
        # -------------------------------------------------

        forward, up = self.apply_euler_rotation(
            forward,
            up,
            controller.local_pitch,
            controller.local_yaw,
            controller.local_roll
        )

        # -------------------------------------------------
        # Actualizar plano
        # -------------------------------------------------

        self.plane.SetOrigin(pos)
        self.plane.SetNormal(up.tolist())

        # -------------------------------------------------
        # Clip + cálculo manual de bounds
        # -------------------------------------------------

        min_x = min_y = min_z = float("inf")
        max_x = max_y = max_z = float("-inf")
        actores_creados = 0

        for clipper, actor in zip(self.clippers, self.actors):
            clipper.Update()
            filled_poly = clipper.GetOutput()

            if filled_poly is None or filled_poly.GetNumberOfCells() == 0:
                actor.SetVisibility(False)
                continue

            actor.SetVisibility(True)
            actores_creados += 1

            b = filled_poly.GetBounds()

            min_x = min(min_x, b[0])
            max_x = max(max_x, b[1])
            min_y = min(min_y, b[2])
            max_y = max(max_y, b[3])
            min_z = min(min_z, b[4])
            max_z = max(max_z, b[5])

        if actores_creados == 0:
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return img, pos, rot

        bounds = (min_x, max_x, min_y, max_y, min_z, max_z)

        dx_b = bounds[1] - bounds[0]
        dy_b = bounds[3] - bounds[2]
        dz_b = bounds[5] - bounds[4]

        diagonal = math.sqrt(dx_b * dx_b + dy_b * dy_b + dz_b * dz_b)
        if diagonal < 1e-6:
            diagonal = 1.0

        # -------------------------------------------------
        # Cámara
        # -------------------------------------------------

        cam_distance = diagonal * 2.0
        cam_pos = pos_np - up * cam_distance

        self.camera.SetPosition(cam_pos.tolist())

        focus_offset = 1
        new_focus = pos_np + forward * focus_offset

        self.camera.SetFocalPoint(new_focus.tolist())
        self.camera.SetViewUp(forward.tolist())
        self.camera.SetParallelProjection(True)

        scale = max(dx_b, dy_b) * 0.6
        if scale < 1e-3:
            scale = 0.5

        self.camera.SetParallelScale(scale)

        self.ren.ResetCameraClippingRange()

        # -------------------------------------------------
        # Render
        # -------------------------------------------------

        self.renWin.Render()

        # -------------------------------------------------
        # Captura imagen
        # -------------------------------------------------

        self.w2if.Modified()
        self.w2if.Update()

        vtk_image = self.w2if.GetOutput()
        dims = vtk_image.GetDimensions()

        vtk_array = vtk_image.GetPointData().GetScalars()
        np_array = numpy_support.vtk_to_numpy(vtk_array)

        img = np_array.reshape(dims[1], dims[0], 3)
        img = np.flipud(img)

        return img.astype(np.uint8), pos, rot
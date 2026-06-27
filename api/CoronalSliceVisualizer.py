import vtk
import numpy as np
from vtk.util import numpy_support

class CoronalSliceVisualizer:


    def __init__(self, meshes, width, height):

        self.meshes = meshes
        self.width = width
        self.height = height

        # off-screen renderer setup
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0, 0, 0)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetOffScreenRendering(1)
        self.render_window.SetSize(width, height)

        # create actors (one per mesh) with empty mappers initially
        self.actors = []  # each item: (actor, original_mesh)
        for mesh, color in meshes:
            mapper = vtk.vtkPolyDataMapper()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color[0], color[1], color[2])
            self.renderer.AddActor(actor)
            self.actors.append((actor, mesh))

        # configure the camera in parallel projection mode
        self.camera = self.renderer.GetActiveCamera()
        self.camera.SetParallelProjection(True)

    def _euler_to_matrix(self, pitch, yaw, roll):

        # rotation around X (pitch)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch),  np.cos(pitch)]])
        # rotation around Y (yaw)
        Ry = np.array([[ np.cos(yaw), 0, np.sin(yaw)],
                       [0, 1, 0],
                       [-np.sin(yaw), 0, np.cos(yaw)]])
        # rotation around Z (roll)
        Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                       [np.sin(roll),  np.cos(roll), 0],
                       [0, 0, 1]])
        return Rz @ Ry @ Rx

    def slice_and_fill_mesh_vtk(self, mesh, origin, normal):

        # cutting plane
        plane = vtk.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)

        # generate the intersection lines
        cutter = vtk.vtkCutter()
        cutter.SetInputData(mesh)
        cutter.SetCutFunction(plane)
        cutter.Update()

        contour_lines = cutter.GetOutput()
        if contour_lines.GetNumberOfCells() == 0:
            # no intersection
            empty = vtk.vtkPolyData()
            return empty

        # fill the closed contours to get a surface
        triangulator = vtk.vtkContourTriangulator()
        triangulator.SetInputData(contour_lines)
        triangulator.Update()
        return triangulator.GetOutput()

    def render_from_controller(self, controller_mov):

        
        position = controller_mov.get_current_position()
        pitch, yaw, roll = controller_mov.get_current_orientation()
        print("Getting position ", position)
        print("Getting orientation (pitch, yaw, roll) in radians: ", (pitch, yaw, roll))
        # build the orthonormal basis
        pitch = np.radians(pitch)   # converts degrees to radians
        yaw   = np.radians(yaw)
        roll  = np.radians(roll)
        R = self._euler_to_matrix(pitch, yaw, roll)

        forward_local = np.array([0, 0, 1])
        right_local   = np.array([1, 0, 0])
        up_local      = np.array([0, 1, 0])

        forward = R @ forward_local
        right   = R @ right_local
        up      = R @ up_local

        # normalize (just in case, even though R is a pure rotation)
        forward = forward / np.linalg.norm(forward)
        right   = right / np.linalg.norm(right)
        up      = up / np.linalg.norm(up)

        offset = -1
        origin = np.array(position) + forward * offset
        normal = up  # normal of the coronal plane

        # list of bounds for visible geometries
        visible_bounds = []
        visible_count = 0

        # slice each mesh and update its actor
        for actor, mesh in self.actors:
            sliced = self.slice_and_fill_mesh_vtk(mesh, origin, normal)
            if sliced.GetNumberOfCells() == 0:
                actor.VisibilityOff()
            else:
                actor.VisibilityOn()
                actor.GetMapper().SetInputData(sliced)
                bounds = sliced.GetBounds()
                # check that the bounds are valid (xmin <= xmax)
                if bounds[1] >= bounds[0]:
                    visible_bounds.append(bounds)
                    visible_count += 1

        # if nothing is visible, return a black image
        if visible_count == 0:
            black_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return black_img, position, (pitch, yaw, roll)

        # combine global bounds
        global_bounds = list(visible_bounds[0])
        for bounds in visible_bounds[1:]:
            for i in range(6):
                if i % 2 == 0:  # minimum
                    global_bounds[i] = min(global_bounds[i], bounds[i])
                else:            # maximum
                    global_bounds[i] = max(global_bounds[i], bounds[i])

        # bounding box diagonal to compute the camera distance
        dx = global_bounds[1] - global_bounds[0]
        dy = global_bounds[3] - global_bounds[2]
        dz = global_bounds[5] - global_bounds[4]
        diagonal = np.sqrt(dx*dx + dy*dy + dz*dz)
        distance = 2.0 * diagonal

        # compute the in-plane extents (right and forward directions)
        # for that we transform the 8 vertices of the bounding box
        # into the local system with axes (right, up, forward)
        corners = []
        for x in (global_bounds[0], global_bounds[1]):
            for y in (global_bounds[2], global_bounds[3]):
                for z in (global_bounds[4], global_bounds[5]):
                    corners.append(np.array([x, y, z]))

        # inverse (transpose) matrix to go from world to local
        R_inv = R.T
        local_corners = [R_inv @ (c - origin) for c in corners]

        # right (local x) and forward (local z) components
        right_vals = [c[0] for c in local_corners]
        forward_vals = [c[2] for c in local_corners]
        width_bb = max(right_vals) - min(right_vals)
        height_bb = max(forward_vals) - min(forward_vals)

        # configure the camera
        self.camera.SetPosition(origin - up * distance)
        self.camera.SetFocalPoint(origin)
        self.camera.SetViewUp(forward)
        self.camera.SetParallelScale(max(width_bb, height_bb) * 0.6)

        # render
        self.render_window.Render()

        # capture image
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(self.render_window)
        w2if.Update()
        vtk_image = w2if.GetOutput()

        # convert to numpy array (we assume 3 RGB channels)
        vtk_array = vtk_image.GetPointData().GetScalars()
        np_image = numpy_support.vtk_to_numpy(vtk_array)
        dims = vtk_image.GetDimensions()  # (width, height, 1)
        np_image = np_image.reshape(dims[1], dims[0], 3)
        np_image = np_image.astype(np.uint8)

        np_image = np.fliplr(np_image)
        return np_image, position, (pitch, yaw, roll)

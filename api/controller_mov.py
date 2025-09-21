import math
import vtk

_controller = None
_bounds = None

def get_bounds():
    global _bounds
    if _bounds is None:
        folder_path = "api/stl-files/0_skin.stl"
        reader = vtk.vtkSTLReader()
        reader.SetFileName(folder_path)
        reader.Update()
        skin_poly_data = reader.GetOutput()
        _bounds = skin_poly_data.GetBounds()
    return _bounds

def init_controller(scale_factor=1.05, y_additional_scale=0.10):
    global _controller
    bounds = get_bounds()
    _controller = EllipsoidMovementController(bounds, scale_factor, y_additional_scale)
    return _controller

def move_transducer(direction):
    global _controller
    if _controller is None:
        init_controller()
    return _controller.move(direction)

def get_current_position():
    global _controller
    if _controller is None:
        init_controller()
    return _controller.calculate_position()

def get_current_angles():
    global _controller
    if _controller is None:
        init_controller()
    return _controller.get_current_angles()

def reset_position():
    global _controller
    if _controller is None:
        init_controller()
    return _controller.reset_position()

class EllipsoidMovementController:
    def __init__(self, bounds, scale_factor=1.05, y_additional_scale=0.10):
        self.bounds = bounds
        self.scale_factor = scale_factor
        self.y_additional_scale = y_additional_scale
        self.x_radius = (bounds[1] - bounds[0]) / 2.0 * scale_factor
        self.y_radius = (bounds[3] - bounds[2]) / 2.0 * (scale_factor + y_additional_scale)
        self.z_radius = (bounds[5] - bounds[4]) / 2.0 * scale_factor
        self.center_x = (bounds[0] + bounds[1]) / 2.0
        self.center_y = (bounds[2] + bounds[3]) / 2.0
        self.center_z = (bounds[4] + bounds[5]) / 2.0
        self.theta = 0
        self.phi = 0
        self.psi = 0
        self.delta_angle = 0.02
        self.delta_angle_psi = 0.05

    def move(self, direction):
        if direction == "left":
            self.theta -= self.delta_angle
        elif direction == "right":
            self.theta += self.delta_angle
        elif direction == "up":
            self.phi += self.delta_angle
        elif direction == "down":
            self.phi -= self.delta_angle
        elif direction == "a":
            self.psi -= self.delta_angle_psi
        elif direction == "d":
            self.psi += self.delta_angle_psi
        else:
            raise ValueError(f"Direccin no vlida: {direction}")
        self.theta = self.theta % (2 * math.pi)
        self.phi = self.phi % (2 * math.pi)
        self.psi = self.psi % (2 * math.pi)
        return self.calculate_position()

    def calculate_position(self):
        x = self.center_x + self.x_radius * math.cos(self.theta) * math.cos(self.phi)
        y = self.center_y + self.y_radius * math.sin(self.theta) * math.cos(self.psi)
        z = self.center_z + self.z_radius * math.sin(self.phi) * math.cos(self.psi)
        return (x, y, z)

    def get_current_angles(self):
        return (math.degrees(self.theta), math.degrees(self.phi), math.degrees(self.psi))

    def reset_position(self):
        self.theta = 0
        self.phi = 0
        self.psi = 0
        return self.calculate_position()
    
    def get_current_position():
        global _controller
        if _controller is None:
            init_controller()
        return _controller.calculate_position()
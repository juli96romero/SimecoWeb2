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
    pos = _controller.move(direction)
    rot = _controller.calculate_orientation()
    return {"position": pos, "rotation": rot}


def rotate_transducer(direction):
    global _controller
    if _controller is None:
        init_controller()
    result = _controller.rotate_in_place(direction)
    return result


def reset_position():
    global _controller
    if _controller is None:
        init_controller()
    return _controller.reset_position()


# =============================
# CLASE PRINCIPAL
# =============================
class EllipsoidMovementController:
    def __init__(self, bounds, scale_factor=1.05, y_additional_scale=0.10):
        self.bounds = bounds
        self.scale_factor = scale_factor
        self.y_additional_scale = y_additional_scale

        # Dimensiones
        self.x_radius = (bounds[1] - bounds[0]) / 2.0 * scale_factor
        self.y_radius = (bounds[3] - bounds[2]) / 2.0 * (scale_factor + y_additional_scale)
        self.z_radius = (bounds[5] - bounds[4]) / 2.0 * scale_factor

        # Centro del cuerpo
        self.center_x = (bounds[0] + bounds[1]) / 2.0
        self.center_y = (bounds[2] + bounds[3]) / 2.0
        self.center_z = (bounds[4] + bounds[5]) / 2.0

        # Ángulos globales (órbita)
        self.theta = 0.0
        self.phi = 0.0

        # Ángulos locales (rotación propia)
        self.local_pitch = 0.0
        self.local_yaw = 0.0
        self.local_roll = 0.0

        # Paso angular
        self.delta_angle = 0.02
        self.delta_local = 0.05

    # --------------------------
    # Movimiento alrededor del cuerpo
    # --------------------------
    def move(self, direction):
        if direction == "left":
            self.theta -= self.delta_angle
        elif direction == "right":
            self.theta += self.delta_angle
        elif direction == "up":
            self.phi += self.delta_angle
        elif direction == "down":
            self.phi -= self.delta_angle
        else:
            raise ValueError(f"Dirección no válida: {direction}")

        self.theta %= 2 * math.pi
        self.phi %= 2 * math.pi

        return self.calculate_position()

    # --------------------------
    # Rotación local (en el lugar)
    # --------------------------
    def rotate_in_place(self, direction):
        if direction == "pitch_up":
            self.local_pitch += self.delta_local
        elif direction == "pitch_down":
            self.local_pitch -= self.delta_local
        elif direction == "yaw_left":
            self.local_yaw -= self.delta_local
        elif direction == "yaw_right":
            self.local_yaw += self.delta_local
        elif direction == "roll_left":
            self.local_roll -= self.delta_local
        elif direction == "roll_right":
            self.local_roll += self.delta_local
        else:
            raise ValueError(f"Dirección de rotación inválida: {direction}")

        rot = self.calculate_orientation()
        pos = self.calculate_position()  # posición no cambia
        return {"position": pos, "rotation": rot}

    # --------------------------
    # Cálculos base
    # --------------------------
    def calculate_position(self):
        x = self.center_x + self.x_radius * math.cos(self.theta) * math.cos(self.phi)
        y = self.center_y + self.y_radius * math.sin(self.theta)
        z = self.center_z + self.z_radius * math.sin(self.phi)
        return (x, y, z)

    def calculate_orientation(self):
        # Direccion hacia el centro (global)
        x, y, z = self.calculate_position()
        dx = self.center_x - x
        dy = self.center_y - y
        dz = self.center_z - z

        # Orientación global
        yaw = math.atan2(dx, dz)
        pitch = math.atan2(dy, math.sqrt(dx**2 + dz**2))

        # Aplicar rotaciones locales
        pitch += self.local_pitch
        yaw += self.local_yaw
        roll = self.local_roll

        return (
            math.degrees(pitch),
            math.degrees(yaw),
            math.degrees(roll)
        )

    def reset_position(self):
        self.theta = self.phi = 0.0
        self.local_pitch = self.local_yaw = self.local_roll = 0.0
        return self.calculate_position()


def get_current_position():
    global _controller
    if _controller is None:
        init_controller()
    return _controller.calculate_position()


def get_current_orientation():
    global _controller
    if _controller is None:
        init_controller()
    return _controller.calculate_orientation()

import math

_controller = None

def get_bounds():
    """Bounds fijos de la malla del cuerpo (skin)"""
    return (-1.01, 1.02, -0.64, 0.63, -0.86, 0.72)

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
    return _controller.rotate_in_place(direction)

def reset_position():
    global _controller
    if _controller is None:
        init_controller()
    return _controller.reset_position()

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


class EllipsoidMovementController:
    def __init__(self, bounds, scale_factor=1.05, y_additional_scale=0.10):
        self.bounds = bounds

        # Centro real del abdomen
        self.center_x = (bounds[0] + bounds[1]) / 2.0
        self.center_y = (bounds[2] + bounds[3]) / 2.0
        self.center_z = (bounds[4] + bounds[5]) / 2.0

        # Radios del elipsoide
        self.x_radius = (bounds[1] - bounds[0]) / 2.0 * scale_factor
        self.y_radius = (bounds[3] - bounds[2]) / 2.0 * (scale_factor + y_additional_scale)
        self.z_radius = (bounds[5] - bounds[4]) / 2.0 * scale_factor

        # Parámetros esféricos
        self.y_offset = 0.0
        self.y_limit = self.y_radius * 0.6  # ajustable           # latitud (altura)
        self.phi = -math.pi / 2         # longitud (arranca en frente +Z)

        # Rotaciones locales
        self.local_pitch = 0.0
        self.local_yaw = 0.0
        self.local_roll = 0.0

        self.delta_angle = 0.02
        self.delta_local = 0.05

    # -------------------------
    # MOVIMIENTO ORBITAL
    # -------------------------

    def move(self, direction):

        if direction == "left":
            self.phi += self.delta_angle

        elif direction == "right":
            self.phi -= self.delta_angle

        elif direction == "up":
            self.y_offset -= self.delta_angle * self.y_radius

        elif direction == "down":
            self.y_offset += self.delta_angle * self.y_radius

        # Clamp
        self.y_offset = max(-self.y_limit, min(self.y_limit, self.y_offset))

        return self.calculate_position()

    # -------------------------
    # ROTACIÓN LOCAL
    # -------------------------

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
            raise ValueError(f"Dirección inválida: {direction}")

        return {
            "position": self.calculate_position(),
            "rotation": self.calculate_orientation()
        }

    # -------------------------
    # POSICIÓN EN EL ELIPSOIDE
    # -------------------------

    def calculate_position(self):

        cos_phi = math.cos(self.phi)
        sin_phi = math.sin(self.phi)

        x = self.center_x + self.x_radius * cos_phi
        z = self.center_z + self.z_radius * sin_phi
        y = self.center_y + self.y_offset

        return (x, y, z)

    # -------------------------
    # ORIENTACIÓN (APUNTA AL CENTRO)
    # -------------------------

    def calculate_orientation(self):

        x, y, z = self.calculate_position()

        # Centro proyectado al mismo Y actual (ignora diferencia vertical)
        dx = self.center_x - x
        dz = self.center_z - z
        dy = 0.0  # ← CLAVE: no mirar hacia arriba/abajo automáticamente

        # yaw alrededor de Y
        yaw = math.atan2(-dx, -dz)

        # pitch solo viene de rotación local
        pitch = 0.0

        pitch += self.local_pitch
        yaw += self.local_yaw
        roll = self.local_roll

        return (
            math.degrees(pitch),
            math.degrees(yaw),
            math.degrees(roll)
        )

    # -------------------------
    # RESET
    # -------------------------

    def reset_position(self):

        self.theta = 0.0
        self.phi = math.pi / 2

        self.local_pitch = 0.0
        self.local_yaw = 0.0
        self.local_roll = 0.0

        return self.calculate_position()

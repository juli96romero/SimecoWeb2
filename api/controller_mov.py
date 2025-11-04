import math
import numpy as np
import vtk

# Controlador global
_controller = None

class TransducerController:
    """
    Controlador simplificado y corregido para el transductor
    """
    
    def __init__(self, bounds, scale_factor=1.2, target_origin=None):
        self.bounds = bounds
        self.scale_factor = scale_factor
        
        # Radios de la elipsoide
        self.x_radius = (bounds[1] - bounds[0]) / 2.0 * scale_factor
        self.y_radius = (bounds[3] - bounds[2]) / 2.0 * scale_factor  
        self.z_radius = (bounds[5] - bounds[4]) / 2.0 * scale_factor
        
        # Centro del cuerpo
        self.center_x = (bounds[0] + bounds[1]) / 2.0
        self.center_y = (bounds[2] + bounds[3]) / 2.0
        self.center_z = (bounds[4] + bounds[5]) / 2.0
        
        # Target origin (centro de la malla)
        if target_origin is None:
            self.target_origin = np.array([self.center_x, self.center_y, self.center_z])
        else:
            self.target_origin = np.array(target_origin)
        
        # Estado del transductor
        self.theta = 0.0  # Ángulo orbital horizontal
        self.phi = 0.0    # Ángulo orbital vertical
        
        # Rotaciones locales (en radianes)
        self.local_pitch = 0.0
        self.local_yaw = 0.0  
        self.local_roll = 0.0
        
        # Parámetros de movimiento
        self.orbital_step = 0.02
        self.rotation_step = 0.05
        
        # Distancia fija del transductor al plano de corte
        self.cut_plane_distance = 1.0
    def calculate_position(self):
        """Calcula posición en la elipsoide"""
        x = self.center_x + self.x_radius * math.cos(self.theta) * math.cos(self.phi)
        y = self.center_y + self.y_radius * math.sin(self.theta)
        z = self.center_z + self.z_radius * math.sin(self.phi)
        return (x, y, z)

    def get_orientation_matrix(self):
        """
        Calcula matriz de orientación total:
        - conserva la orientación base para mirar al target (como antes),
        - aplica rotaciones locales en el marco del transductor (multiplicación por R_local).
        """
        # Posición y vector hacia target
        pos = np.array(self.calculate_position())
        to_target = self.target_origin - pos
        to_target /= np.linalg.norm(to_target)

        # Calculamos yaw_base/pitch_base exactamente como antes (para mantener la pose inicial)
        yaw_base = math.atan2(to_target[0], to_target[2])
        pitch_base = math.asin(-to_target[1])
        roll_base = 0.0

        # Construimos R_target usando la misma convención Yaw-Pitch-Roll (como tenías originalmente)
        cy, sy = math.cos(yaw_base), math.sin(yaw_base)
        cp, sp = math.cos(pitch_base), math.sin(pitch_base)
        cr, sr = math.cos(roll_base), math.sin(roll_base)

        R_target = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr]
        ])

        # Ahora matriz local construida a partir de los angulos locales (aplicada en el marco local)
        cp_l, sp_l = math.cos(self.local_pitch), math.sin(self.local_pitch)
        cy_l, sy_l = math.cos(self.local_yaw), math.sin(self.local_yaw)
        cr_l, sr_l = math.cos(self.local_roll), math.sin(self.local_roll)

        R_local = np.array([
            [cy_l * cp_l, cy_l * sp_l * sr_l - sy_l * cr_l, cy_l * sp_l * cr_l + sy_l * sr_l],
            [sy_l * cp_l, sy_l * sp_l * sr_l + cy_l * cr_l, sy_l * sp_l * cr_l - cy_l * sr_l],
            [-sp_l,       cp_l * sr_l,                     cp_l * cr_l]
        ])

        # Composición: primero mirar al target (pose inicial), luego aplicar rotaciones locales en el marco del transductor
        R_total = R_target @ R_local
        return R_total

    def get_forward_direction(self):
        """Dirección forward del transductor (-Y local)"""
        R = self.get_orientation_matrix()
        return R @ np.array([0, -1, 0])

    def get_cut_plane(self):
        """
        Calcula el plano de corte correctamente:
        - Origin: punto en el espacio donde está el plano
        - Normal: dirección en la que mira el transductor (hacia la malla)
        """
        transductor_pos = np.array(self.calculate_position())
        forward_dir = self.get_forward_direction()
        
        # El plano está a una distancia fija del transductor en dirección forward
        plane_origin = transductor_pos + forward_dir * self.cut_plane_distance
        
        # La normal del plano es opuesta a la dirección forward (apunta hacia el transductor)
        plane_normal = -forward_dir
        
        return plane_origin.tolist(), plane_normal.tolist()

    def move(self, direction):
        """Movimiento orbital"""
        if direction == "left":
            self.theta -= self.orbital_step
        elif direction == "right":
            self.theta += self.orbital_step
        elif direction == "up":
            self.phi += self.orbital_step
        elif direction == "down":
            self.phi -= self.orbital_step
            
        self.theta %= 2 * math.pi
        self.phi %= 2 * math.pi
        
        return self.calculate_position()

    def rotate(self, direction):
        """Rotación local in-place"""
        if direction == "pitch_up":
            self.local_pitch += self.rotation_step
        elif direction == "pitch_down":
            self.local_pitch -= self.rotation_step
        elif direction == "yaw_left":
            self.local_yaw -= self.rotation_step
        elif direction == "yaw_right":
            self.local_yaw += self.rotation_step
        elif direction == "roll_left":
            self.local_roll -= self.rotation_step
        elif direction == "roll_right":
            self.local_roll += self.rotation_step
            
        return self.get_orientation_degrees()

    def get_orientation_degrees(self):
        """Orientación en grados para el frontend"""
        R = self.get_orientation_matrix()
        
        # Extraer ángulos de Euler de la matriz
        pitch = math.asin(-R[2, 0])
        yaw = math.atan2(R[1, 0], R[0, 0])
        roll = math.atan2(R[2, 1], R[2, 2])
        
        return (math.degrees(pitch), math.degrees(yaw), math.degrees(roll))

    def reset(self):
        """Reset a posición y rotación inicial"""
        self.theta = 0.0
        self.phi = 0.0
        self.local_pitch = 0.0
        self.local_yaw = 0.0
        self.local_roll = 0.0
        return self.calculate_position()

# =============================================================================
# API Pública (compatible con tu código existente)
# =============================================================================

def init_controller(scale_factor=1.2, target_origin=None):
    global _controller
    bounds = get_bounds()
    _controller = TransducerController(bounds, scale_factor=scale_factor, target_origin=target_origin)
    return _controller

def _orientation_payload(controller):
    R = controller.get_orientation_matrix()
    forward = (R @ np.array([0, -1, 0])).tolist()  # forward world (punta del transductor), -Y local como vos definiste
    R_flat = R.flatten().tolist()  # row-major
    eulers = controller.get_orientation_degrees()
    return {"rotation_matrix": R_flat, "forward": forward, "eulers": eulers}

def move_transducer(direction):
    global _controller
    if _controller is None:
        init_controller()
    pos = _controller.move(direction)
    payload = _orientation_payload(_controller)
    return {"position": pos, **payload}

def rotate_transducer(direction):
    global _controller
    if _controller is None:
        init_controller()
    _controller.rotate(direction)
    pos = _controller.calculate_position()
    payload = _orientation_payload(_controller)
    return {"position": pos, **payload}

# get_current_* similar:
def get_current_position():
    global _controller
    if _controller is None:
        init_controller()
    pos = _controller.calculate_position()
    payload = _orientation_payload(_controller)
    return {"position": pos, **payload}

def reset_position():
    global _controller
    if _controller is None:
        init_controller()
    return _controller.reset()


def get_current_orientation():
    global _controller
    if _controller is None:
        init_controller()
    return _controller.get_orientation_degrees()

def get_current_cut_plane():
    global _controller
    if _controller is None:
        init_controller()
    return _controller.get_cut_plane()

def get_target_origin():
    global _controller
    if _controller is None:
        init_controller()
    return _controller.target_origin.tolist()

# Función auxiliar para obtener bounds (mantener tu existente)
def get_bounds():
    folder_path = "api/stl_para usar/0_skin.stl"
    reader = vtk.vtkSTLReader()
    reader.SetFileName(folder_path)
    reader.Update()
    skin_poly_data = reader.GetOutput()
    return skin_poly_data.GetBounds()
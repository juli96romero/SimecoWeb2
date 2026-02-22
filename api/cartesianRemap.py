import numpy as np
import cv2
import pickle
from polarTransform.imageTransform import ImageTransform

class FOVOptimizer:
    def __init__(self):
        self.rotation_matrix = None
        self.superior_mask = None
        self.inferior_mask = None
        self.cartesian_map_x = None
        self.cartesian_map_y = None
        self._precomputed = False
    
    def precompute_for_128x128(self, pickle_path='api/mask_v1.pickle'):
        """Pre-computa TODO el pipeline para imgenes 128x128"""
        print("Pre-computando pipeline completo para 128x128...")
        
        # Cargar la configuracin existente del pickle
        with open(pickle_path, 'rb') as f:
            existing_pt = pickle.load(f)
        
        # 1. Pre-computar rotacin para 128x128
        height, width = 128, 128
        center = ((height-1)/2.0, (width-1)/2.0)
        self.rotation_matrix = cv2.getRotationMatrix2D(center, 90, 1)
        
        # 2. Mscaras (igual que antes)
        self.superior_mask = np.zeros((333, 612, 3), dtype=np.uint8)
        self.inferior_mask = np.zeros((103, 612, 3), dtype=np.uint8)
        
        # 3. Pre-computar el mapeo cartesiano para polarTransform 2.0.0
        # Crear los puntos de la imagen cartesiana de destino
        cartesian_height, cartesian_width = existing_pt.cartesianImageSize
        y_cart, x_cart = np.indices((cartesian_height, cartesian_width))
        
        # Crear array de puntos [x, y] para la transformacin
        points = np.stack([x_cart, y_cart], axis=-1).reshape(-1, 2)
        
        # Usar el objeto existente para calcular los puntos polares
        polar_points = existing_pt.getPolarPointsImage(points)
        
        # Reformatear a la forma original
        polar_points = polar_points.reshape(cartesian_height, cartesian_width, 2)
        
        # Separar en mapas x e y para cv2.remap
        self.cartesian_map_x = polar_points[:, :, 0].astype(np.float32)
        self.cartesian_map_y = polar_points[:, :, 1].astype(np.float32)
        
        self._precomputed = True
        print("Pipeline completo pre-computado ")
        print(f"Tamao del mapeo cartesiano: {self.cartesian_map_x.shape}")
        print(f"Rango mapa X: {self.cartesian_map_x.min():.1f} to {self.cartesian_map_x.max():.1f}")
        print(f"Rango mapa Y: {self.cartesian_map_y.min():.1f} to {self.cartesian_map_y.max():.1f}")

# Instancia global - precomputar AL INICIAR la aplicacin
fov_optimizer = FOVOptimizer()

# IMPORTANTE: Llama esta funcin UNA SOLA VEZ al iniciar tu aplicacin
fov_optimizer.precompute_for_128x128()

def acomodarFOV_ultra_rapido(img):
    """
    Versin ultra optimizada que usa precomputado
    Entrada: imagen 128x128x3
    Salida: imagen transformada
    """
    # Verificar que est precomputado
    if not fov_optimizer._precomputed:
        raise RuntimeError("FOVOptimizer no ha sido precomputado. Llama precompute_for_128x128() primero.")
    
    # 1. Rotacin (precomputada)
    rotatedImage = cv2.warpAffine(img, fov_optimizer.rotation_matrix, (128, 128))
    
    # 2. Scaling 
    scaledImage = cv2.resize(rotatedImage, (612, 203), interpolation=cv2.INTER_AREA)
    
    # 3. Concatenacin (precomputada)
    conc_1 = cv2.vconcat([fov_optimizer.superior_mask, scaledImage])
    conc_2 = cv2.vconcat([conc_1, fov_optimizer.inferior_mask])
    
    # 4. Transformacin cartesiana (ULTRA RPIDA - precomputada)
    reconstructedImage = cv2.remap(conc_2, 
                                 fov_optimizer.cartesian_map_x, 
                                 fov_optimizer.cartesian_map_y,
                                 interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
    
    return reconstructedImage
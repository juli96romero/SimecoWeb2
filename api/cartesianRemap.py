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
        """Precompute the pipeline for 128x128 images"""
        print("Precomputing the full pipeline for 128x128...")
        
        # load the existing configuration from the pickle
        with open(pickle_path, 'rb') as f:
            existing_pt = pickle.load(f)
        
        # 1. precompute the rotation for 128x128
        height, width = 128, 128
        center = ((height-1)/2.0, (width-1)/2.0)
        self.rotation_matrix = cv2.getRotationMatrix2D(center, 90, 1)
        
        # 2. masks
        self.superior_mask = np.zeros((333, 612, 3), dtype=np.uint8)
        self.inferior_mask = np.zeros((103, 612, 3), dtype=np.uint8)
        
        # 3. precompute the cartesian map for polarTransform 2.0.0
        # build the points of the destination cartesian image
        cartesian_height, cartesian_width = existing_pt.cartesianImageSize
        y_cart, x_cart = np.indices((cartesian_height, cartesian_width))
        
        # build the [x, y] points array for the transform
        points = np.stack([x_cart, y_cart], axis=-1).reshape(-1, 2)
        
        # use the existing object to compute the polar points
        polar_points = existing_pt.getPolarPointsImage(points)
        
        # reshape to the original form
        polar_points = polar_points.reshape(cartesian_height, cartesian_width, 2)
        
        # split into x and y maps for cv2.remap
        self.cartesian_map_x = polar_points[:, :, 0].astype(np.float32)
        self.cartesian_map_y = polar_points[:, :, 1].astype(np.float32)
        
        self._precomputed = True
        print("Full pipeline precomputed")
        print(f"Cartesian map size: {self.cartesian_map_x.shape}")
        print(f"Map X range: {self.cartesian_map_x.min():.1f} to {self.cartesian_map_x.max():.1f}")
        print(f"Map Y range: {self.cartesian_map_y.min():.1f} to {self.cartesian_map_y.max():.1f}")

# global instance, precomputed once when the module is imported
fov_optimizer = FOVOptimizer()
fov_optimizer.precompute_for_128x128()

def apply_fov_remap(image):
    """
    Input: 128x128x3 image
    Output: image transformed into the transducer field of view (FOV)
    """
    if not fov_optimizer._precomputed:
        raise RuntimeError("FOVOptimizer has not been precomputed. Call precompute_for_128x128() first.")

    # 1. rotation
    rotated_image = cv2.warpAffine(image, fov_optimizer.rotation_matrix, (128, 128))

    # 2. scaling
    scaled_image = cv2.resize(rotated_image, (612, 203), interpolation=cv2.INTER_AREA)

    # 3. concatenation
    with_top_mask = cv2.vconcat([fov_optimizer.superior_mask, scaled_image])
    with_bottom_mask = cv2.vconcat([with_top_mask, fov_optimizer.inferior_mask])

    # 4. cartesian transform
    cartesian_image = cv2.remap(with_bottom_mask,
                                fov_optimizer.cartesian_map_x,
                                fov_optimizer.cartesian_map_y,
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))

    return cartesian_image

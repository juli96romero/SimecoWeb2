import cv2
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance

class BitStreamOptimizer:
    def __init__(self):
        # Pre-asignar buffer para JPEG encoding
        self.jpeg_buffer = bytearray(200 * 1024)  # 200KB buffer
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # Calidad balanceada
    
    def formatAsBitStream_optimized(self, image_data):
        """
        Versin optimizada: OpenCV + JPEG + buffer pre-asignado
        """
        # Asegurar tipo de datos
        if image_data.dtype != np.uint8:
            image_data = image_data.astype(np.uint8)
        
        # ENCODING CON OPENCV (MUCHO ms rpido que PIL + PNG)
        success, encoded_image = cv2.imencode('.jpg', image_data, self.encode_param)
        
        if success:
            # Convertir a base64
            image_base64 = base64.b64encode(encoded_image).decode('utf-8')
            return image_base64
        else:
            # Fallback a la versin original si falla
            return self.formatAsBitStream_original(image_data)
    
    def formatAsBitStream_original(self, image_data):
        """Tu versin original por si acaso"""
        image_data = image_data.astype(np.uint8)
        height, width, channels = image_data.shape
        image_data_reshaped = image_data.reshape((height, width, channels))
        
        image = Image.fromarray(image_data_reshaped)
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()


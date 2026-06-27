import cv2
import base64
import numpy as np
from io import BytesIO
from PIL import Image

class BitStreamOptimizer:
    def __init__(self):
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

    def formatAsBitStream_optimized(self, image_data):
        if image_data.dtype != np.uint8:
            image_data = image_data.astype(np.uint8)

        success, encoded_image = cv2.imencode('.jpg', image_data, self.encode_param)

        if success:
            image_base64 = base64.b64encode(encoded_image).decode('utf-8')
            return image_base64
        else:
            return self.formatAsBitStream_original(image_data)

    def formatAsBitStream_original(self, image_data):
        image_data = image_data.astype(np.uint8)
        height, width, channels = image_data.shape
        image_data_reshaped = image_data.reshape((height, width, channels))

        image = Image.fromarray(image_data_reshaped)
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()

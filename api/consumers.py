import json
from channels.generic.websocket import WebsocketConsumer
from .red import main  
import base64
import numpy as np

from io import BytesIO
from PIL import Image

input_path = "./data/validation/labels"
output_path = "./results/" 

class ChatConsumer(WebsocketConsumer):

    model = main("self")
    def connect(self):
        self.accept()
        
        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected'
        }))

    def receive(self, text_data):
        # Assuming you receive image data in base64 format
        
        image_data = self.model.short_valid_step(input_path,output_path)
        
        # Convert image data to uint8
        image_data = image_data.astype(np.uint8)
        
        # Reshape the image data to (height, width, channels)
        height, width, channels = image_data.shape
        image_data_reshaped = image_data.reshape((height, width, channels))

        # Convert the image array to PIL Image
        image = Image.fromarray(image_data_reshaped)

        # Convert the image to base64
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

        self.send(text_data=json.dumps({"image_data": image_base64}))

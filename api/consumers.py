import json
from channels.generic.websocket import WebsocketConsumer
from .red import main  
import base64
import numpy as np
from . import views
from .cartesianRemap import acomodarFOV
from .cartesianRemap import pickel_images
from io import BytesIO
from PIL import Image
from os import path
import os



from .red_copy import main as main2

input_path = "./data/validation/labels"
output_path = "./results/" 
model = main("self")



class ChatConsumer(WebsocketConsumer):

    
    def connect(self):
        self.accept()
        
        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected'
        }))

    def receive(self, text_data):
        # Assuming you receive image data in base64 format
        
        imagen_recorte_vtk = views.vtk_visualization_image(text_data)
        
        output_path = "./results/imagenesTomadasDeVTK/"
        os.makedirs(output_path, exist_ok=True)
        image = Image.fromarray(imagen_recorte_vtk.astype('uint8'), 'RGB')
        image.save(os.path.join(output_path, 'consumers1.png'))

        image_data = model.valid_step256_fromImage(img_generada=imagen_recorte_vtk)
        
        # Convert image data to uint8
        image_data = image_data.astype(np.uint8)
        
        

        # Reshape the image data to (height, width, channels)
        height, width, channels = image_data.shape
        image_data_reshaped = image_data.reshape((height, width, channels))

        

        # Convert the image array to PIL Image
        image = Image.fromarray(image_data_reshaped)


        #image = acomodarFOV(img=image)
        # Convert the image to base64
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
        #"######################################################################################################################"
        #parte agregada  ACA########################################################################################################################
        imagen_recorte_vtk = imagen_recorte_vtk.astype(np.uint8)
        
        

        # Reshape the image data to (height, width, channels)
        height, width, channels = imagen_recorte_vtk.shape
        image_data_reshaped = imagen_recorte_vtk.reshape((height, width, channels))

        

        # Convert the image array to PIL Image
        image = Image.fromarray(image_data_reshaped)


        #image = acomodarFOV(img=image)
        # Convert the image to base64
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            image_base64Nueva = base64.b64encode(buffer.getvalue()).decode()



        self.send(text_data=json.dumps({"image_data": image_base64, "new_image_data": image_base64Nueva}))


class ImageConsumer(WebsocketConsumer):

    
    def connect(self):
        self.accept()
        
        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected'
        }))

    def receive(self, text_data):
        # Assuming you receive image data in base64 format
        
        image_data = views.vtk_visualization_image(text_data)
        
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


class Outputer(WebsocketConsumer):

    
    def connect(self):
        self.accept()
        
        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected'
        }))

    def receive(self, text_data):
        # Assuming you receive image data in base64 format
        input_path = "./data/validation/labels"
        output_path = "./results/"
        model.validation_step(input_path,output_path)
        
        # Convert image data to uint8
        pickel_images("asd")

        self.send(text_data=json.dumps({"image_data": "a"}))



class GeneradorLineal(WebsocketConsumer):

    
    def connect(self):
        self.accept()
        
        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected'
        }))

    def receive(self, text_data):
        # Assuming you receive image data in base64 format
        
        image_data = model.short_valid_step_256(input_path,output_path)
        
        # Convert image data to uint8
        image_data = image_data.astype(np.uint8)
        
        

        # Reshape the image data to (height, width, channels)
        height, width, channels = image_data.shape
        image_data_reshaped = image_data.reshape((height, width, channels))

        

        # Convert the image array to PIL Image
        image = Image.fromarray(image_data_reshaped)


        #image = acomodarFOV(img=image)
        # Convert the image to base64
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

        self.send(text_data=json.dumps({"image_data": image_base64}))



class Principal(WebsocketConsumer):

    model2 = main2("self")
    def connect(self):
        self.accept()
        
        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected'
        }))

    def receive(self, text_data):
        # Assuming you receive image data in base64 format
        
        image_data = self.model2.short_valid_step(input_path,output_path)
        
        # Convert image data to uint8
        image_data = image_data.astype(np.uint8)
        
        

        # Reshape the image data to (height, width, channels)
        height, width, channels = image_data.shape
        image_data_reshaped = image_data.reshape((height, width, channels))

        

        # Convert the image array to PIL Image
        image = Image.fromarray(image_data_reshaped)


        #image = acomodarFOV(img=image)
        # Convert the image to base64
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

        self.send(text_data=json.dumps({"image_data": image_base64}))


class Principal128(WebsocketConsumer):

    
    def connect(self):
        self.accept()
        
        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected'
        }))

    def receive(self, text_data):
        # Assuming you receive image data in base64 format
        
        image_data = model.short_valid_step(input_path,output_path)
        
        # Convert image data to uint8
        image_data = image_data.astype(np.uint8)
        
        

        # Reshape the image data to (height, width, channels)
        height, width, channels = image_data.shape
        image_data_reshaped = image_data.reshape((height, width, channels))

        

        # Convert the image array to PIL Image
        image = Image.fromarray(image_data_reshaped)


        #image = acomodarFOV(img=image)
        # Convert the image to base64
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

        self.send(text_data=json.dumps({"image_data": image_base64}))
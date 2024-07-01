import json
from channels.generic.websocket import WebsocketConsumer
from .red import main  
import base64
import numpy as np
from . import views
from .cartesianRemap import acomodarFOV
from io import BytesIO
from PIL import Image
import os
from os import path, listdir
from albumentations.pytorch import ToTensorV2
import cv2
from .red_2 import main as main2

input_path = "./data/validation/labels"
output_path = "./results/" 
#Levanto las dos redes
model = main("self")
model2 = main2("self")

class Socket_Principal_FrontEnd(WebsocketConsumer):
    
    def connect(self):
        self.accept()
        
        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected'
        }))

    def receive(self, text_data):

        #genero la imagen con VTK para usar como label 
        imagen_recorte_vtk = views.vtk_visualization_image(text_data)
        
        #le mando la imagen al model para que haga una inferencia y me la devuelva
        image_data = model.valid_step256_fromImage(img_generada=imagen_recorte_vtk)

        #convierto esa imagen a cartesianas
        image_data = acomodarFOV(image_data)

        image_base64= formatAsBitStream(image_data=image_data)
        
        self.send(text_data=json.dumps({"image_data": image_base64}))



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
        image_base64= formatAsBitStream(image_data=image_data)

        self.send(text_data=json.dumps({"image_data": image_base64}))


class PickleHandler(WebsocketConsumer):

    indice =0
    #model.validation_step(input_path,output_path)
    def connect(self):
        self.accept()
        
        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected'
        }))

    def receive(self, text_data):
        # Assuming you receive image data in base64 format
        
        output_path = "./results/"
        filenames = listdir(output_path)
        
        if len(filenames)>self.indice:
            indice = 0
        
        image = cv2.imread(path.join(output_path, filenames[self.indice]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        indice+=1
        # Convert image data to uint8
        image_data = acomodarFOV(image)

        image_base64= formatAsBitStream(image_data=image_data)

        self.send(text_data=json.dumps({"image_data": image_base64}))



class Principal128(WebsocketConsumer):
    indice = 0
    
    def connect(self):
        self.accept()
        
        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected'
        }))

    def receive(self, text_data):

        input_path = "./data/validation/labels"

        filenames = listdir(input_path)

        image = cv2.imread(path.join(input_path, filenames[self.indice]))
        # By default OpenCV uses BGR color space, we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.indice+=1
        if self.indice>=len(filenames):
            self.indice=0
        image_data = model.valid_step256_fromImage(img_generada=image)
        
        # Convert image data to uint8
        image_base64 = formatAsBitStream(image_data=image_data)

        self.send(text_data=json.dumps({"image_data": image_base64}))



class Principal256(WebsocketConsumer):
    indice = 0
    
    def connect(self):
        self.accept()
        
        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected'
        }))

    def receive(self, text_data):
        # Assuming you receive image data in base64 format
        
        input_path = "./data/validation/labels"

        filenames = listdir(input_path)

        image = cv2.imread(path.join(input_path, filenames[self.indice]))
        # By default OpenCV uses BGR color space, we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.indice+=1
        if self.indice>=len(filenames):
            self.indice=0
        image_data = model2.valid_step256_fromImage(img_generada=image)
        
        image_base64 = formatAsBitStream(image_data=image_data)
        
        self.send(text_data=json.dumps({"image_data": image_base64}))

def formatAsBitStream(image_data):
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
    return image_base64
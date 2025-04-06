import json
from channels.generic.websocket import WebsocketConsumer
from .red import main  
import base64
import numpy as np
from . import views
from .cartesianRemap import acomodarFOV
from io import BytesIO
from PIL import Image, ImageEnhance
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

brillo = [0, 0, 0, 0, 0, 0, 0, 0, 0]


class Socket_Principal_FrontEnd(WebsocketConsumer):
    brillo = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    new_position= (0,0,0)
    direction = None

    def connect(self):
        self.accept()

        

        
        #views.vtk_visualization(None,{0,0,0})
        
        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected'
        }))

    def receive(self, text_data):

        views.update_normal(text_data)
        #genero la imagen con VTK para usar como label 
        imagen_recorte_vtk = views.vtk_visualization_image(text_data)
        
        #le mando la imagen al model para que haga una inferencia y me la devuelva
        image_data = model.hacerInferencia(img_generada=imagen_recorte_vtk)

        data = json.loads(text_data)
        print(data)

        try:
            # Asegúrate de convertir el valor a entero
            data = json.loads(text_data)
            self.brillo[0] = int(data.get('brightness'))
            self.brillo[1] = int(data.get('brightness1'))
            self.brillo[2] = int(data.get('brightness2'))
            self.brillo[3] = int(data.get('brightness3'))
            self.brillo[4] = int(data.get('brightness4'))
            self.brillo[5] = int(data.get('brightness5'))
            self.brillo[6] = int(data.get('brightness6'))
            self.brillo[7] = int(data.get('brightness7'))
            self.brillo[8] = int(data.get('brightness8'))
            self.direction = data.get('direction')
        except (ValueError, TypeError):
            print("El valor de brillo no es válido. Usando el valor por defecto")

        if self.direction:
        # Mover el transductor según la dirección
            print(self.new_position)
            print("mobverreeriorajfñfafarfrñjfa")
            self.new_position = views.move_transducer(self.direction)
            print(self.new_position)
        else:
            print("no moveeeemento")
        
        imagen_brillo_ajustado = ajustar_brillo_con_franjas(image_data,self.brillo)
        #convierto esa imagen a cartesianas
        image_fov = acomodarFOV(imagen_brillo_ajustado)

        #PARA GUARDAR LOS ARCHIVOS
        #filenames = listdir(output_path)
        #output_filename = os.path.join(output_path, filenames[0] + '.png')
        #image = Image.fromarray(image_data)
        #image.save(output_filename)
        
        image_base64= formatAsBitStream(image_data=image_fov)
        
        

        self.send(text_data=json.dumps({"image_data": image_base64, "position": self.new_position}))


    def adjust_brightness(self, image_base64, brightness_factor):
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        
        enhancer = ImageEnhance.Brightness(image)
        adjusted_image = enhancer.enhance(brightness_factor)

        buffered = BytesIO()
        adjusted_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    

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

    indice = 0
    
    def connect(self):
        self.accept()
        
        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected'
        }))

    def receive(self, text_data):
        # Assuming you receive image data in base64 format
        
        output_path = "./results"
        filenames = listdir(output_path)
        
        if self.indice>=len(filenames):
            self.indice = 1
        
        print(path.join(output_path, filenames[self.indice]))
        image = cv2.imread(path.join(output_path, filenames[self.indice]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.uint8)

        self.indice+=1
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
        image_data = model.hacerInferencia(img_generada=image)

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
        image_data = model2.hacerInferencia(img_generada=image)
        
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

class Prueba(WebsocketConsumer):
    
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
        image_data = model.hacerInferencia(img_generada=imagen_recorte_vtk)

        #convierto esa imagen a cartesianas
        image_data = acomodarFOV(image_data)

        image_base64= formatAsBitStream(image_data=image_data)
        
        self.send(text_data=json.dumps({"image_data": image_base64}))

class Prueba2(WebsocketConsumer):
    
    def connect(self):
        self.accept()
        
        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected'
        }))

    def receive(self, text_data):

        #genero la imagen con VTK para usar como label 
        
        full_image, subimage, mask_image = views.generate_subimage_with_fov(text_data)
        
        #le mando la imagen al model para que haga una inferencia y me la devuelva
        #image_data = model.hacerInferencia(img_generada=imagen_recorte_vtk)

        #convierto esa imagen a cartesianas
        #image_data = acomodarFOV(image_data)

        full_image= formatAsBitStream(image_data=full_image)
        #subimage= formatAsBitStream(image_data=full_image)
        #mask_image= formatAsBitStream(image_data=full_image)

        
        
        self.send(text_data=json.dumps({"image_data": full_image,"image_data2": subimage,"image_data3": mask_image}))

class Brightness(WebsocketConsumer):
    indice = 0
    brillo = [0, 0, 0, 0, 0]
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
        image_data = model.hacerInferencia(img_generada=image)

        try:
            # Asegúrate de convertir el valor a entero
            data = json.loads(text_data)
            self.brillo[0] = int(data.get('brightness', 50))
            self.brillo[1] = int(data.get('brightness1', 50))
            self.brillo[2] = int(data.get('brightness2', 50))
            self.brillo[3] = int(data.get('brightness3', 50))
            self.brillo[4] = int(data.get('brightness4', 50))
        except (ValueError, TypeError):
            print("El valor de brillo no es válido. Usando el valor por defecto")

        imagen_brillo_ajustado = ajustar_brillo_con_franjas(image_data,self.brillo)

        # Convert image data to uint8
        image_base64 = formatAsBitStream(image_data=imagen_brillo_ajustado)

        self.send(text_data=json.dumps({"image_data": image_base64}))


def ajustar_brillo_con_franjas(imagen, brillo):
    print("brillo:", brillo)
    
    # Convertir la imagen a tipo int16 para evitar desbordamiento en las operaciones
    imagen_float = imagen.astype(np.int16)

    # Aplicar el ajuste general de brillo
    imagen_ajustada = imagen_float + brillo[0]

    # Calcular el número de filas y determinar las franjas
    filas, columnas = imagen.shape[:2]
    franja_altura = filas // 8  # Ahora tenemos 8 franjas

    # Aplicar ajustes de brillo específicos por franja
    for i in range(8):
        inicio = i * franja_altura
        fin = (i + 1) * franja_altura if i < 7 else filas  # La última franja llega hasta el final
        imagen_ajustada[inicio:fin, :] += brillo[i + 1]  # brillo[1] a brillo[8]

    # Clip para mantener los valores dentro del rango [0, 255]
    imagen_ajustada = np.clip(imagen_ajustada, 0, 255).astype(np.uint8)

    return imagen_ajustada
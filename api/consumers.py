import json
from channels.generic.websocket import WebsocketConsumer

from api.fastVtkVisualizer import FastVtkVisualizer
from .red import main  
import base64
import pickle

import numpy as np
from . import views
from .cartesianRemap import acomodarFOV_ultra_rapido
from io import BytesIO
from PIL import Image, ImageEnhance
import os
from os import path, listdir
import cv2
from .red_2 import main as main2
from . import controller_mov as mov
from . import fastVtkVisualizer as vtk_visualizer
import time
import logging
import time
import logging
import json
from channels.generic.websocket import WebsocketConsumer
from api.cartesianRemap import fov_optimizer  
import polarTransform
from .bitstream_optimizer import BitStreamOptimizer
input_path = "./data/validation/labels"
output_path = "./results/" 
#Levanto las dos redes
model = main("self")
model2 = main2("self")

brillo = [0, 0, 0, 0, 0, 0, 0, 0, 0]
bitstream_optimizer = BitStreamOptimizer()


# Configurar logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler("performance.log"),
                       logging.StreamHandler()
                   ])

class Socket_Principal_FrontEnd(WebsocketConsumer):
    brillo = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    direction = None
    _last_processing_time = 0
    _total_requests = 0
    _total_processing_time = 0
    mallas = None

    def connect(self):


        with open('api/mask_v1.pickle', 'rb') as f:
            ptSettings = pickle.load(f)

        self.accept()

        if not self.mallas:
            self.mallas = views.getMallas()

        # 🔥 CREAR MOTOR VTK UNA SOLA VEZ
        if not hasattr(self, "vtk_engine"):
            self.vtk_engine = FastVtkVisualizer(self.mallas, width=300, height=300)

        fov_optimizer.precompute_for_128x128()

        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected'
        }))
    def receive(self, text_data):
        start_time = time.time()
        self._total_requests += 1
        
        try:
            parse_start = time.time()
            data = json.loads(text_data)
            parse_time = time.time() - parse_start
            logging.info(f"Parse time: {parse_time*1000:.2f}ms")

            # Extraccin de datos
            extract_start = time.time()
            self.direction = data.get('direction')
            special_position = data.get('specialActorPosition')
            
            # Extraer brillos
            for i in range(9):
                brightness_key = f'brightness{i}' if i > 0 else 'brightness'
                self.brillo[i] = int(data.get(brightness_key, 0))
            extract_time = time.time() - extract_start
            logging.info(f"Data extraction time: {extract_time*1000:.2f}ms")

            # Movimiento del transductor
            move_time = 0
            nueva_posicion = mov.get_current_position()
            nueva_rotacion = mov.get_current_orientation()
            
            if self.direction == "reset":
                reset_start = time.time()
                nueva_posicion = mov.reset_position()
                move_time = time.time() - reset_start
                logging.info(f"Reset time: {move_time*1000:.2f}ms")
            elif self.direction:
                move_start = time.time()
                if data["action"] == "move":
                    nueva_posicion = mov.move_transducer(self.direction)
                elif data["action"] == "rotate":
                    nueva_rotacion = mov.rotate_transducer(self.direction)
                move_time = time.time() - move_start
                logging.info(f"Move time: {move_time*1000:.2f}ms")

            # Procesamiento VTK 
            vtk_time = 0
            if not self.direction:  # Solo procesar VTK si no hay movimiento
                vtk_start = time.time()

                imagen_recorte_vtk, pos, rot = self.vtk_engine.vtk_visualization_images(mov, image_rotation_deg=0)

                vtk_time = time.time() - vtk_start
                logging.info(f"VTK processing time: {vtk_time*1000:.2f}ms")
            else:
                imagen_recorte_vtk = None

            # Inferencia de red neuronal
            inference_time = 0
            if imagen_recorte_vtk is not None:
                inference_start = time.time()
                image_data = model.hacerInferencia(img_generada=imagen_recorte_vtk)
                inference_time = time.time() - inference_start
                logging.info(f"Inference time: {inference_time*1000:.2f}ms")
            else:
                image_data = None

            # Procesamiento posterior de imagen
            process_time = 0
            image_base64 = None
            if image_data is not None:
                process_start = time.time()

                # Ajuste de brillo
                brightness_start = time.time()
                imagen_brillo_ajustado = ajustar_brillo_con_franjas(image_data, self.brillo)
                brightness_time = time.time() - brightness_start
                logging.info(f"  Brightness adjustment: {brightness_time*1000:.2f}ms")

                # Acomodar FOV
                fov_start = time.time()
                image_fov = acomodarFOV_ultra_rapido(imagen_brillo_ajustado)
                fov_time = time.time() - fov_start
                logging.info(f"  FOV processing: {fov_time*1000:.2f}ms")

                # Formatear a bitstream
                bitstream_start = time.time()
                image_base64 = bitstream_optimizer.formatAsBitStream_optimized(image_fov)
                bitstream_time = time.time() - bitstream_start
                logging.info(f"  Bitstream conversion: {bitstream_time*1000:.2f}ms")

                process_time = time.time() - process_start

            # Tiempo total
            total_time = time.time() - start_time
            self._total_processing_time += total_time
            
            # Log del rendimiento
            logging.info(f"TOTAL REQUEST TIME: {total_time*1000:.2f}ms")
            logging.info(f"Average processing time: {self._total_processing_time/self._total_requests*1000:.2f}ms")
            logging.info(f"Estimated FPS: {1/total_time if total_time > 0 else 0:.1f}")
            logging.info("-" * 50)
            # Enviar respuesta
            response_data = {
                "image_data": image_base64,
                "image_data_2": bitstream_optimizer.formatAsBitStream_optimized(imagen_recorte_vtk),
                "position": nueva_posicion,
                "processing_time": total_time,
                "direction": self.direction,
                "rotation": nueva_rotacion
            }
            
            self.send(text_data=json.dumps(response_data))

        except Exception as e:
            error_time = time.time() - start_time
            logging.error(f"Error after {error_time*1000:.2f}ms: {str(e)}")
            self.send(text_data=json.dumps({"error": str(e)}))


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
        
        image_data = views.vtk_visualization_image(text_data)
        # Convert image data to uint8
        image_base64= formatAsBitStream(image_data=image_data)

        self.send(text_data=json.dumps({"image_data": image_base64}))


class PickleHandler(WebsocketConsumer):

    indice = 0
    
    def connect(self):
        import pickle

        # Cargar el archivo pickle de forma segura
        with open('api/mask_v1.pickle', 'rb') as f:
            ptSettings = pickle.load(f)

        self.accept()
        
        # Pre-computar transformaciones FOV una sola vez para mejorar el rendimiento
        fov_optimizer.precompute_for_128x128()
        
        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected'
        }))

    def receive(self, text_data):
        
        input_path = "./results"
        filenames = listdir(input_path)
        
        if self.indice>=len(filenames):
            self.indice = 0
        image = cv2.imread(path.join(input_path, filenames[self.indice]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.uint8)

        image_data = acomodarFOV_ultra_rapido(image)

        image_base64= formatAsBitStream(image_data=image_data)

        self.send(text_data=json.dumps({"image_data": image_base64}))



class Principal128(WebsocketConsumer): #RV Visualizacion basada en imagenes de la carpeta
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.indice+=1
        if self.indice>=len(filenames):
            self.indice=0
        image_data = model.hacerInferencia(img_generada=image)
        image_base64 = formatAsBitStream(image_data=image_data)

        self.send(text_data=json.dumps({"image_data": image_base64}))



class Principal256(WebsocketConsumer):#RV Visualizacion basada en imagenes de la carpeta con la red de 256px
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.indice+=1
        if self.indice>=len(filenames):
            self.indice=0
        image_data = model2.hacerInferencia(img_generada=image)
        
        image_base64 = formatAsBitStream(image_data=image_data)
        
        self.send(text_data=json.dumps({"image_data": image_base64}))

def formatAsBitStream(image_data): #RV
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
    return image_base64

class combinedSlice(WebsocketConsumer):
    
    def connect(self):
        import pickle

        # Cargar el archivo pickle de forma segura
        with open('api/mask_v1.pickle', 'rb') as f:
            ptSettings = pickle.load(f)

        self.accept()
        
        # Pre-computar transformaciones FOV una sola vez para mejorar el rendimiento
        fov_optimizer.precompute_for_128x128()
        
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
        image_data = acomodarFOV_ultra_rapido(image_data)

        image_base64= formatAsBitStream(image_data=image_data)
        
        self.send(text_data=json.dumps({"image_data": image_base64}))


class Brightness(WebsocketConsumer):
    indice = 0
    brillo = [0, 0, 0, 0, 0, 0, 0, 0, 0]
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
            # Asegrate de convertir el valor a entero
            data = json.loads(text_data)
            self.brillo[0] = int(data.get('brightness', 50))
            self.brillo[1] = int(data.get('brightness1', 50))
            self.brillo[2] = int(data.get('brightness2', 50))
            self.brillo[3] = int(data.get('brightness3', 50))
            self.brillo[4] = int(data.get('brightness4', 50))
        except (ValueError, TypeError):
            print("El valor de brillo no es valido. Usando el valor por defecto")

        imagen_brillo_ajustado = ajustar_brillo_con_franjas(image_data,self.brillo)

        # Convert image data to uint8
        image_base64 = formatAsBitStream(image_data=imagen_brillo_ajustado)

        self.send(text_data=json.dumps({"image_data": image_base64}))


def ajustar_brillo_con_franjas(imagen, brillo):
    
    # Convertir la imagen a tipo int16 para evitar desbordamiento en las operaciones
    imagen_float = imagen.astype(np.int16)

    # Aplicar el ajuste general de brillo
    imagen_ajustada = imagen_float + brillo[0]

    # Calcular el nmero de filas y determinar las franjas
    filas, columnas = imagen.shape[:2]
    franja_altura = filas // 8  # Ahora tenemos 8 franjas

    # Aplicar ajustes de brillo especficos por franja
    for i in range(8):
        inicio = i * franja_altura
        fin = (i + 1) * franja_altura if i < 7 else filas  # La ltima franja llega hasta el final
        imagen_ajustada[inicio:fin, :] += brillo[i + 1]  # brillo[1] a brillo[8]

    # Clip para mantener los valores dentro del rango [0, 255]
    imagen_ajustada = np.clip(imagen_ajustada, 0, 255).astype(np.uint8)

    return imagen_ajustada
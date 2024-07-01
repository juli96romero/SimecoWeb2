import math
import cv2
import numpy as np
import polarTransform
import matplotlib.pyplot as plt
import pickle
from os import scandir, getcwd

def scale_image(image, width, height):
    """ Retorna la imagen re-escalada"""
    scaledImage = cv2.resize(image,(width,height), interpolation=cv2.INTER_AREA) 
    return scaledImage   

def ls(ruta = getcwd()):
    """ Levantar los archivos de una carpeta """
    return [arch.name for arch in scandir(ruta) if arch.is_file()]

input_path = './results/' 
output_path = './resizedimages/'
#pickle_path =  '../resizedimages/mask_simeco.pickle'
pickle_path =  'api/mask_v1.pickle'  
# Load a .pkl file
pickle_file = open(pickle_path,'rb')
ptSettings = pickle.load(pickle_file)
pickle_file.close()

def pickel_images(self):
    lista_img = ls(input_path)

    for image_file in lista_img:
        
        
        # Open image
        img = cv2.imread(input_path + image_file)
        width = img.shape[1]
        height = img.shape[0]
        # 1ero tengo que restaurar la imagen original para coincidir con las propiedades de las coordenadas polares guardadas en el archivo pickle
        m2 = cv2.getRotationMatrix2D(((height-1)/2.0,(width-1)/2.0),90,1)
        rotatedImage = cv2.warpAffine(img,m2,(height,width))
        scaledImage = scale_image(rotatedImage, 612, 203)  
        superior = np.zeros((333,612,3), dtype=scaledImage.dtype)
        inferior = np.zeros((103,612,3), dtype=scaledImage.dtype)
        conc_1 = cv2.vconcat([superior,scaledImage])
        conc_2 = cv2.vconcat([conc_1,inferior])
        print(conc_2.shape)
        # Conc_2 deberias tener una dimension de (639, 612, 3)
        reconstructedImage = ptSettings.convertToCartesianImage(conc_2)
        # Guardamos la imagen   
        cv2.imwrite(output_path +"rec_" + image_file,reconstructedImage)

def acomodarFOV(img):
    width = img.shape[1]
    height = img.shape[0]
    # 1ero tengo que restaurar la imagen original para coincidir con las propiedades de las coordenadas polares guardadas en el archivo pickle
    m2 = cv2.getRotationMatrix2D(((height-1)/2.0,(width-1)/2.0),90,1)
    rotatedImage = cv2.warpAffine(img,m2,(height,width))
    scaledImage = scale_image(rotatedImage, 612, 203)  
    superior = np.zeros((333,612,3), dtype=scaledImage.dtype)
    inferior = np.zeros((103,612,3), dtype=scaledImage.dtype)
    conc_1 = cv2.vconcat([superior,scaledImage])
    conc_2 = cv2.vconcat([conc_1,inferior])
    
    # Conc_2 deberias tener una dimension de (639, 612, 3)
    reconstructedImage = ptSettings.convertToCartesianImage(conc_2)
    # Guardamos la imagen   
    

    return reconstructedImage

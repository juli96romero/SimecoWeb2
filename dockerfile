# Utiliza una imagen base oficial de Python
FROM python:3.9-slim

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia los archivos de la aplicación a /app
COPY . /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0
# Instala las dependencias directamente
RUN pip install django daphne channels numpy pillow matplotlib opencv-python tensorboard pytorch-lightning torchvision albumentations polarTransform vtk

# Exponer el puerto que la aplicación usará (si es necesario)
EXPOSE 8000

# Define el comando por defecto para ejecutar la aplicación
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

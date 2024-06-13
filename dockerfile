# Utiliza una imagen base oficial de Python
FROM python:3.9-slim

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia los archivos de la aplicación a /app
COPY . /app

# Instala las dependencias que no están en requirements.txt
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1

# Instala las dependencias adicionales que no están en requirements.txt
RUN pip install django daphne channels numpy pillow matplotlib opencv-python tensorboard pytorch-lightning torchvision albumentations polarTransform vtk

# Copia el archivo de requerimientos
COPY requirements.txt .

# Instala las dependencias listadas en requirements.txt
RUN pip install -r requirements.txt

# Instala el paquete ca-certificates del sistema operativo
RUN apt-get install -y ca-certificates

# Exponer el puerto que la aplicación usará (si es necesario)
EXPOSE 8000

# Define el comando por defecto para ejecutar la aplicación
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

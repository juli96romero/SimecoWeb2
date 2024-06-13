# SimecoWEB

## Iniciar el Proyecto con Conda

Pasos para levantar el proyecto desde el ambiente Conda:

1. **Descargar el repositorio SimecoWEB:**

   ```sh
   git clone https://github.com/juli96romero/SimecoWeb2.git

3. **Descargar los datos:**

   Descargar los datos desde [SimecoWEB] (https://drive.google.com/file/d/1sW3vYTLPcXwhtjWcs70pv8qPrhxsh7GA/view?usp=drive_link)
   Descomprimir y pegar dentro de SimecoWEB

4. **Crear y activar ambiente de Conda:**

   ```sh
   conda create --name env_simeco
   conda activate env_simeco

5. **Instalar dependencias:**

   ```sh
   pip install django
   pip install daphne
   pip install channels
   pip install numpy
   pip install pillow
   pip install matplotlib
   pip install opencv-python
   pip install tensorboard
   pip install pytorch-lightning
   pip install torchvision
   pip install albumentations
   pip install polarTransform 
   pip install vtk


6. **Correr el servidor:**

   ```sh
   python manage.py runserver

6. **Acceder via ip (indicada en la terminal):**

   [Acceso a Simeco WEB vía localhost](http://127.0.0.1:8000/)

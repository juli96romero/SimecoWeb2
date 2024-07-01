# SimecoWEB

## Iniciar el Proyecto con Conda

Pasos para levantar el proyecto desde el ambiente Conda:

1. **Descargar el repositorio SimecoWEB:**

   ```sh
   git clone https://github.com/juli96romero/SimecoWeb2.git

3. **Descargar los datos:**

   Descargar los datos desde [Datos de SIMECO](https://drive.google.com/file/d/1sW3vYTLPcXwhtjWcs70pv8qPrhxsh7GA/view?usp=drive_link).
   
   Descargar el archivo CKTP desde [Archivo CKTP](https://drive.google.com/file/d/1XKMFaIcEyTFAhf_wGqB2w8VLbaU6xqRq/view?usp=drive_link).

   Descomprimir y pegar dentro de la carpeta SimecoWEB

5. **Crear y activar ambiente de Conda:**

   ```sh
   conda create --name env_simeco
   conda activate env_simeco
   conda install pip

6. **Instalar dependencias:**

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

6.5. **Instalar dependencias extras para LINUX:**

   ```sh
   sudo apt update
   sudo apt install mesa-utils
   sudo apt install -y mesa-utils
   sudo apt install libxrender1
   sudo apt install libxtst6 libxrandr2 libxcb1 libxext6
   sudo apt install xorg


7. **Correr el servidor:**

   ```sh
   python manage.py runserver

6. **Acceder via ip (indicada en la terminal):**

   [Acceso a Simeco WEB vía localhost](http://127.0.0.1:8000/)

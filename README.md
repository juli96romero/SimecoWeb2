# SimecoWEB

## Iniciar el Proyecto con Conda

Pasos para levantar el proyecto desde el ambiente Conda:

1. **Descargar el repositorio SimecoWEB:**

   ```sh
   git clone https://github.com/juli96romero/SimecoWeb2.git

2. **Descargar los datos:**

   Descargar los datos desde [Datos de SIMECO](https://drive.google.com/file/d/1VihFXJlI73ICX5GMJvQGRBGRUaN1pn1c/view?usp=drive_link)
   
   Descomprimir y pegar dentro de la carpeta SimecoWEB

3. **Crear y activar ambiente de Conda:**

   ```sh
   conda create --name env_simeco
   conda activate env_simeco
   conda install pip

4. **Instalar dependencias:**

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

5. **Correr el servidor:**

   ```sh
   python manage.py runserver

6. **((Opcional)) Instalar dependencias extras en caso de que algo falle:**

   ```sh
   pip install -r requirements.txt

7. **Acceder via ip (indicada en la terminal):**

   [Acceso a Simeco WEB vía localhost](http://127.0.0.1:8000/)

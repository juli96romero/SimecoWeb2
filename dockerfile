FROM continuumio/miniconda3

# Copiar el archivo de entorno conda exportado
COPY environment.yml .

# Crear un nuevo entorno conda y activarlo
RUN pip install
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

# Copiar el código de tu proyecto al contenedor
COPY . /app
WORKDIR /app



# Comando predeterminado para ejecutar cuando se inicie el contenedor
CMD ["python", "app.py"]
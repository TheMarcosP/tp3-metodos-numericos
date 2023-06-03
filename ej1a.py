import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Directorio que contiene las imágenes
directorio_imagenes = 'dataset_imagenes'

# Lista para almacenar los vectores de las imágenes
vectores_imagenes = []

# Obtener la lista de archivos en el directorio
archivos = os.listdir(directorio_imagenes)

# Iterar sobre los archivos del directorio
for archivo in archivos:
    # Comprobar si el archivo es una imagen
    if archivo.endswith('.png') or archivo.endswith('.jpg') or archivo.endswith('.jpeg'):
        # Leer la imagen utilizando la biblioteca PIL
        ruta_imagen = os.path.join(directorio_imagenes, archivo)
        imagen = Image.open(ruta_imagen)
        
        # Convertir la imagen a un arreglo de NumPy
        arreglo_imagen = np.array(imagen)
        
        # Apilar el arreglo de la imagen como un vector y añadirlo a la lista
        vector_imagen = arreglo_imagen.flatten()
        vectores_imagenes.append(vector_imagen)

# Apilar los vectores de imágenes en una matriz de NumPy
matriz_imagenes = np.column_stack(vectores_imagenes)

# Realizar la descomposición SVD de la matriz
U, S, V = np.linalg.svd(matriz_imagenes)

# Número de primeras y últimas dimensiones a visualizar
p = 30

# Visualizar las primeras dimensiones (autovectores)
plt.figure(figsize=(10, 4))
for i in range(p):
    plt.subplot(2, p, i+1)
    autovector = U[:, i].reshape(arreglo_imagen.shape)
    plt.imshow(autovector, cmap='gray')
    plt.axis('off')
    plt.title(f'Autovector {i+1}')

# Visualizar las últimas dimensiones (autovectores)
for i in range(p):
    plt.subplot(2, p, p+i+1)
    autovector = U[:, -i-1].reshape(arreglo_imagen.shape)
    plt.imshow(autovector, cmap='gray')
    plt.axis('off')
    plt.title(f'Autovector {matriz_imagenes.shape[1]-i}')

plt.tight_layout()
plt.show()

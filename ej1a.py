import numpy as np
import os
from PIL import Image

directorio_imagenes = "dataset_imagenes"

# Obtener la lista de archivos en el directorio
lista_archivos = os.listdir(directorio_imagenes)

vectores_imagenes = []

# Recorrer cada archivo de imagen en el directorio
for archivo in lista_archivos:
    ruta_imagen = os.path.join(directorio_imagenes, archivo)
    imagen = Image.open(ruta_imagen)

    # Convertir la imagen en una matriz NumPy
    matriz_imagen = np.array(imagen)

    p = matriz_imagen.shape[0]  # asumiendo que la imagen es cuadrada (p x p)
    
    # Transformar la imagen en un vector
    vector_imagen = matriz_imagen.reshape(p*p)
    
    vectores_imagenes.append(vector_imagen)

# Convertir la lista de vectores en un arreglo de NumPy
vectores_imagenes = np.array(vectores_imagenes)

U, S, Vt = np.linalg.svd(vectores_imagenes)

#plot in semilog S
# import matplotlib.pyplot as plt
# plt.semilogy(S)

# plt.plot(S)
# plt.show()

# now compress the images
k = 16
# keep only the first k singular values
S = S[:k]
# keep only the first k columns of U
U = U[:,:k]
# keep only the first k rows of Vt
Vt = Vt[:k,:]

# reconstruct the images
for i in range(len(lista_archivos)):
    vector_imagen = vectores_imagenes[i,:]
    # project the image vector onto the first k singular vectors
    vector_imagen_comprimido = np.dot(vector_imagen, Vt.T)
    # reconstruct
    vector_imagen_reconstruido = np.dot(vector_imagen_comprimido, Vt)
    matriz_imagen_reconstruido = vector_imagen_reconstruido.reshape(p,p)

    imagen_reconstruido = Image.fromarray(matriz_imagen_reconstruido)
    imagen_reconstruido = imagen_reconstruido.convert("RGB")
    imagen_reconstruido.save("imagen_reconstruido_" + str(i) + ".png")


# keep the last k singular values
k = 10
# keep only the last k singular values
S = S[-k:]
# keep only the first k columns of U
U = U[:,:k]
# keep only the first k rows of Vt
Vt = Vt[-k:,:]

# reconstruct the images
for i in range(len(lista_archivos)):
    vector_imagen = vectores_imagenes[i,:]
    # project the image vector onto the last k singular vectors
    vector_imagen_comprimido = np.dot(vector_imagen, Vt.T)
    # reconstruct
    vector_imagen_reconstruido = np.dot(vector_imagen_comprimido, Vt)
    matriz_imagen_reconstruido = vector_imagen_reconstruido.reshape(p,p)

    imagen_reconstruido = Image.fromarray(matriz_imagen_reconstruido)
    imagen_reconstruido = imagen_reconstruido.convert("RGB")
    imagen_reconstruido.save("imagen_reconstruido_ultimos" + str(i) + ".png")











# pca es hacer svd y ver las columnas de esa matriz
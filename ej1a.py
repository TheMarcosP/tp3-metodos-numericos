import numpy as np
import os
from PIL import Image

directorio_imagenes = "dataset_imagenes"

# Obtener la lista de archivos en el directorio
lista_archivos = os.listdir(directorio_imagenes)

lista_vectores_imgs = []

# Recorrer cada archivo de imagen en el directorio
for i, archivo in enumerate(lista_archivos):
    ruta_imagen = os.path.join(directorio_imagenes, archivo)
    imagen = Image.open(ruta_imagen)

    # Convertir la imagen en una matriz NumPy
    matriz_imagen = np.asarray(imagen)

    p = matriz_imagen.shape[0]  # asumiendo que la imagen es cuadrada (p x p)
    
    # Transformar la imagen en un vector
    vector_imagen = matriz_imagen.reshape(p*p)

    # crea un vector de largo p*p con i en cada posicion
    # vector_imagen = np.full(p*p, i)
    
    lista_vectores_imgs.append(vector_imagen)


matriz = np.array(lista_vectores_imgs).T


A1 = matriz[:, 4]

# print(A1)
# print(A1.shape)

# reshape as p*p
A1 = A1.reshape(p, p)

A1 = Image.fromarray(A1)
A1.save("pruebaZZ"+ ".png")



# # print(matriz)
# # print(matriz.shape)



U, S, Vt = np.linalg.svd(matriz)
# print(U.shape, S.shape, Vt.shape)

# # guardar reconstruccion de la imagen



# # # keep the last k singular values
# # k = 16
# # # keep only the first k singular values

# # S = S[:k]
# # # keep only the first k columns of U
# # U = U[:, :k]
# # # keep only the first k rows of Vt
# # Vt = Vt[:k, :]
# # # reconstruct the matrix

# # matriz_reconstruida = U.dot(np.diag(S)).dot(Vt)

# # # for i in range(16):
# # #     # get the i-th column of the reconstructed matrix
# # #     imagen_reconstruida = matriz_reconstruida[:, i]
# # #     # reshape the vector to a matrix
# # #     imagen_reconstruida = imagen_reconstruida.reshape(p, p)

# # #     imagen_reconstruido = Image.fromarray(imagen_reconstruida)
# # #     imagen_reconstruido = imagen_reconstruido.convert("RGB")
# # #     imagen_reconstruido.save("aare" + str(i) + ".png")


# # guardar las Visualizar en forma matricial p×p las primeras y las últimas dimensiones (autovectores) de la descomposición
# # obtenida.

# # primer columna de U

for i in range(18):
    U1 = U[i,:]

    # # print(U1)
    # # print(U1.shape)

    # # reshape as p*p
    U1 = U1.reshape(p, p)

    from PIL import ImageOps

    U1 = U1 - np.min(U1)  # Asegurar valores no negativos
    U1 = U1 / np.max(U1)  # Normalizar entre 0 y 1
    U1 = U1 * 255  # Escalar al rango 0-255
    U1 = U1.astype(np.uint8)  # Convertir a tipo entero sin signo de 8 bits

    U1_image = Image.fromarray(U1)
    # U1_image = ImageOps.invert(U1_image)  # Invertir los colores de la imagen
    U1_image.save(f"pruebaU{i}.png")
# # pca es hacer svd y ver las columnas de esa matriz
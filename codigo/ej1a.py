#%%
import numpy as np
import os
from PIL import Image

directorio_imagenes = "dataset_imagenes"

lista_archivos = os.listdir(directorio_imagenes)

lista_vectores_imgs = []

for i, archivo in enumerate(lista_archivos):
    ruta_imagen = os.path.join(directorio_imagenes, archivo)
    imagen = Image.open(ruta_imagen)

    # Convertir la imagen en una matriz NumPy
    matriz_imagen = np.asarray(imagen)

    p = matriz_imagen.shape[0]  # asumiendo que la imagen es cuadrada (p x p)
    
    # Transformar la imagen en un vector
    vector_imagen = matriz_imagen.reshape(p*p)

    # crea un vector de largo p*p con i en cada posicion para debuguear
    # vector_imagen = np.full(p*p, i)
    
    lista_vectores_imgs.append(vector_imagen)


matriz = np.array(lista_vectores_imgs).T


# A1 = matriz[:, 4]

print(matriz)
print(matriz.shape)

# # reshape as p*p
# A1 = A1.reshape(p, p)

# A1 = Image.fromarray(A1)
# A1.save("pruebaZZ"+ ".png")



# # print(matriz)
# # print(matriz.shape)



U, S, Vt = np.linalg.svd(matriz)
print(U.shape, S.shape, Vt.shape)

#prnt max and min of each matrix
print("U max: ", np.max(U))
print("U min: ", np.min(U))
print("S max: ", np.max(S))
print("S min: ", np.min(S))
print("Vt max: ", np.max(Vt))
print("Vt min: ", np.min(Vt))

#%%

# # # guardar reconstruccion de la imagen
# # keep the last k singular values
# k = 16
# # keep only the first k singular values

# S = S[:k]
# # keep only the first k columns of U
# U = U[:, :k]
# # keep only the first k rows of Vt
# Vt = Vt[:k, :]
# # reconstruct the matrix

# print(U.shape, S.shape, Vt.shape)

# matriz_reconstruida = U.dot(np.diag(S))
# print("U dot S:",matriz_reconstruida.shape)
# matriz_reconstruida = matriz_reconstruida.dot(Vt)
# print(matriz_reconstruida.shape)

# for i in range(16):
#     # get the i-th column of the reconstructed matrix
#     imagen_reconstruida = matriz_reconstruida[:, i]
#     # reshape the vector to a matrix
#     imagen_reconstruida = imagen_reconstruida.reshape(p, p)

#     imagen_reconstruido = Image.fromarray(imagen_reconstruida)
#     imagen_reconstruido = imagen_reconstruido.convert("RGB")
#     imagen_reconstruido.save("aare" + str(i) + ".png")






# # guardar las Visualizar en forma matricial p×p las primeras y las últimas dimensiones (autovectores) de la descomposición
# # obtenida.

#%%
# primeras columna de U
# print("u shape: ", U.shape)

# for i in range(16):
#     UI = U[:,i]

#     # U1 dot S1

#     # U1.dot(S[i])

#     # print(U1)
#     print("u1 shape: ", UI.shape)

#     # # reshape as p*p
#     UI = UI.reshape(p, p)

#     from PIL import ImageOps

#     UI = UI - np.min(UI)  # Asegurar valores no negativos
#     UI = UI / np.max(UI)  # Normalizar entre 0 y 1
#     UI = UI * 255  # Escalar al rango 0-255
#     UI = UI.astype(np.uint8)  # Convertir a tipo entero sin signo de 8 bits

#     U1_image = Image.fromarray(UI)
#     # U1_image = ImageOps.invert(U1_image)  # Invertir los colores de la imagen
#     U1_image.save(f"OGcolumna{i}.pdf", "PDF")
# # # pca es hacer svd y ver las columnas de esa matriz


# print("u shape: ", U.shape)


#primeras filas
# for i in range(18):
#     UI = U[i,:]


#     # print(U1)
#     print("u1 shape: ", UI.shape)

#     # # reshape as p*p
#     UI = UI.reshape(p, p)

#     from PIL import ImageOps

#     UI = UI - np.min(UI)  # Asegurar valores no negativos
#     UI = UI / np.max(UI)  # Normalizar entre 0 y 1
#     UI = UI * 255  # Escalar al rango 0-255
#     UI = UI.astype(np.uint8)  # Convertir a tipo entero sin signo de 8 bits

#     U1_image = Image.fromarray(UI)
#     # U1_image = ImageOps.invert(U1_image)  # Invertir los colores de la imagen
#     U1_image.save(f"fila{i}.pdf", "PDF")
# # pca es hacer svd y ver las columnas de esa matriz

#%%


# # subplot de las primeras 16 columnas de U
# import numpy as np
# import matplotlib.pyplot as plt

# def guardar_subplot(matriz):
#     # Crear una figura y un subplot de 4x4
#     fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    
#     # Recorrer las primeras 16 columnas de la matriz
#     for i in range(1,17):
#         # Obtener la subimagen de 28x28 píxeles
#         UI = matriz[:, i]
#         UI = UI.reshape(p, p)
        
#         # Mostrar la subimagen en el subplot correspondiente
#         ax = axs[i//4, i%4]
#         ax.imshow(UI, cmap='gray')
#         ax.axis('off')
    
#     # Ajustar los espacios entre subplots
#     plt.tight_layout()
    
#     # Guardar la figura en un archivo
#     plt.savefig('subplot.png')
    
#     # Mostrar la figura en pantalla
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('tp03.mplstyle')


def guardar_subplot(matriz, num_imagenes, rango_columnas):
    # Calcular el número de filas necesarias para mostrar todas las imágenes
    num_filas = (num_imagenes - 1) // 4 + 1
    
    # Crear una figura y un subplot según el número de imágenes
    fig, axs = plt.subplots(num_filas, 4, figsize=(8, 2*num_filas))
    
    # Recorrer las columnas dentro del rango especificado
    for i, columna in enumerate(range(rango_columnas[0], rango_columnas[1])):
        # Verificar si se han mostrado todas las imágenes deseadas
        if i == num_imagenes:
            break
        
        # Obtener la subimagen de 28x28 píxeles
                # Obtener la subimagen de 28x28 píxeles
        UI = matriz[:, columna]
        UI = UI.reshape(p, p)
        
        # Mostrar la subimagen en el subplot correspondiente
        ax = axs[i//4, i%4]
        ax.imshow(UI, cmap='gray')
        ax.axis('off')
    
    # Ajustar los espacios entre subplots
    plt.tight_layout()
    
    # Guardar la figura en un archivo
    plt.savefig('primerascolumnasU.pdf', format='pdf', dpi=1200)
    
    # Mostrar la figura en pantalla
    # plt.show()
    plt.close()


# guardar_subplot(U,20, (0,20))

# plt.style.use('tp03.mplstyle')
# plt.style.use('seaborn-whitegrid')
#plot singular values

plt.plot(S, label="valores singulares")
# plt.semilogy()
plt.xlabel('cantidad de valores singulares')
plt.ylabel('valor')
#add label to the plot

# add a legend
plt.legend()


plt.savefig('singularvalues.pdf', format='pdf', dpi=1200)
# plt.show()
plt.close()

# suma de los valores singulares
suma = np.sum(S)
print("suma: ", suma)
#calcula la suma acumulada de los valores singulares
suma_acumulada = np.cumsum(S)
# la suma acumulada es la suma de los valores singulares, y la diferencia 
print("suma acumulada: ", suma_acumulada)

print("suma acumulada: ", suma_acumulada/suma * 100)


plot = plt.plot(suma_acumulada/suma * 100)
plt.savefig('suma_acumulada.pdf', format='pdf', dpi=1200)
# %%

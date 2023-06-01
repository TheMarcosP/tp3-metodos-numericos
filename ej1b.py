import numpy as np
from PIL import Image

def compress_image(filepath, k):
    imagen = Image.open(filepath)

    # Convertir la imagen en una matriz NumPy
    matriz_imagen = np.array(imagen)

    U, S, VT = np.linalg.svd(matriz_imagen)

    Sd = np.diag(S)
    comprimida = U[:,:k] @ Sd[0:k,:k] @ VT[:k,:]
    return comprimida



def frobenius_norm(A):
    return np.linalg.norm(A, ord='fro')


file_path = 'dataset_imagenes\img08.jpeg'

# Carga las imágenes originales y comprimidas
img_original = Image.open(file_path)

# Convierte las imágenes en matrices NumPy
mat_original = np.array(img_original)

min = 1
max = img_original.size[0]
for k in range(min,max):
    mat_comprimida = compress_image( file_path,k)
    img_comprimida = Image.fromarray(mat_comprimida)
    img_comprimida = img_comprimida.convert("RGB")
    # img_comprimida.save('img_comprimida.png')
    if mat_original.shape != mat_comprimida.shape:
        mat_comprimida = mat_comprimida.resize(mat_original.shape)
    error = frobenius_norm(mat_original - mat_comprimida) / frobenius_norm(mat_original)
    print(f"con k = {k}",'Error de compresión: {:.2%}'.format(error))
    if error < 0.05:
        break

img_comprimida.save('img_comprimida.png')

# . ¿Qué error obtienen si realizan la misma compresión (con el mismo d) para otra imagen
# cualquiera del conjunto?
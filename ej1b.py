import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


plt.style.use('tp03.mplstyle')

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

# hace un subfigure de la imagen original y la comprimida y la guarda en formato pdf
def compare_compresion_save(mat_comprimida,img_original, filename):
    img_comprimida = Image.fromarray(mat_comprimida)
    img_comprimida = img_comprimida.convert("RGB")
    img_comprimida.save('img_comprimida.png')

    img_original = img_original.convert("RGB")

    # plot the compressed image and the original image in RGB
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img_original)
    axs[0].set_title('Original')
    axs[1].imshow(img_comprimida)
    axs[1].set_title('Comprimida')

    # save as pdf image
    plt.savefig(filename, format='pdf', dpi=1200)
    plt.close()

# encuentra el k para el cual el error de compresión es menor a error
def reduce_error_compresion(filepath, error):
    img_original = Image.open(filepath)

    # Convierte las imágenes en matrices NumPy
    mat_original = np.array(img_original)

    min = 1
    max = img_original.size[0]
    for k in range(min,max):
        mat_comprimida = compress_image(filepath, k)
        img_comprimida = Image.fromarray(mat_comprimida)
        img_comprimida = img_comprimida.convert("RGB")
        if mat_original.shape != mat_comprimida.shape:
            mat_comprimida = mat_comprimida.resize(mat_original.shape)
        error_comprimida = frobenius_norm(mat_original - mat_comprimida) / frobenius_norm(mat_original)
        print(f"con k = {k}", 'Error de compresión: {:.2%}'.format(error_comprimida))
        if error_comprimida < error:
            break
    return k 

def plot_compresion_lost_hist(filepath, k=10, savename = 'img_comprimida.pdf'):
    valores = []

    img_original = Image.open(filepath)
    # Convierte las imágenes en matrices NumPy
    mat_original = np.array(img_original)

    min_k = 0
    max_k = k

    for _k in range(min_k, max_k + 1):
        mat_comprimida = compress_image(filepath, _k)
        img_comprimida = Image.fromarray(mat_comprimida)
        img_comprimida = img_comprimida.convert("RGB")
        
        if mat_original.shape != mat_comprimida.shape:
            mat_comprimida = mat_comprimida.resize(mat_original.shape)
        
        error_comprimida = frobenius_norm(mat_original - mat_comprimida) / frobenius_norm(mat_original)
        valores.append(error_comprimida*100)
        print(f"con k = {_k}", 'Error de compresión: {:.2%}'.format(error_comprimida))

    etiquetas = [str(i) for i in range(min_k, max_k + 1)]

    plt.bar(etiquetas, valores)
    plt.semilogy()
    plt.axhline(y=5, color='r', linestyle='--', label='5% de error')
    plt.legend()

    # Modificar para mostrar solo valores pares en el eje x
    xticks = np.arange(min_k, max_k + 1, 2)
    plt.xticks(xticks, [str(i) for i in xticks])

    plt.xlabel('cantidad de valores singulares')
    plt.ylabel('error de compresión en %')
    # plt.title('Error de compresión en función de la cantidad de valores singulares')
    plt.savefig(savename, format='pdf', dpi=1200)
    plt.close()

#plot all the images in the folder as a subplot
import os
import matplotlib.pyplot as plt

def plot_images_from_folder(folder_path, num_columns=4, column_space=0.1, save_path=None):
    images = []
    
    # Obtain the list of file names in the folder
    file_names = os.listdir(folder_path)
    
    # Read the images and store them in a list
    for file_name in file_names:
        image_path = os.path.join(folder_path, file_name)
        image = plt.imread(image_path)
        images.append(image)
    
    num_images = len(images)
    num_rows = (num_images + num_columns - 1) // num_columns
    
    # Create the subfigure and plot the images
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 12))
    
    for i, ax in enumerate(axes.flatten()):
        if i < num_images:
            ax.imshow(images[i], cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')
    
    # Adjust the space between the columns
    plt.subplots_adjust(wspace=column_space)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_compresion_lost_mean(folderpath, k=10, savename='img_comprimida.pdf'):

    # get all the files in the folder
    import os
    files = os.listdir(folderpath)

    valores_totales = []
    # unos ["img03.jpeg", "img06.jpeg", "img08.jpeg", "img14.jpeg"]
    for file in files:
        valores = []
        filepath = folderpath + file


        img_original = Image.open(filepath)
        # Convierte las imágenes en matrices NumPy
        mat_original = np.array(img_original)

        min_k = 0
        max_k = k

        for _k in range(min_k, max_k + 1):
            mat_comprimida = compress_image(filepath, _k)
            img_comprimida = Image.fromarray(mat_comprimida)
            img_comprimida = img_comprimida.convert("RGB")
            
            if mat_original.shape != mat_comprimida.shape:
                mat_comprimida = mat_comprimida.resize(mat_original.shape)
            
            error_comprimida = frobenius_norm(mat_original - mat_comprimida) / frobenius_norm(mat_original)
            # if error_comprimida*100 < 0.01:
            #     error_comprimida = 0.00105
            valores.append(error_comprimida*100)
            print(f"con k = {_k}", 'Error de compresión: {:.2%}'.format(error_comprimida))

        etiquetas = [str(i) for i in range(min_k, max_k + 1)]

        valores_totales.append(valores)
        plt.plot(etiquetas, valores, linestyle="dashed")
        # modifica la linea para que sea gris
        plt.gca().lines[-1].set_color("gray")
        plt.semilogy()

        # Modificar para mostrar solo valores pares en el eje x
        xticks = np.arange(min_k, max_k + 1, 2)
        plt.xticks(xticks, [str(i) for i in xticks])

        plt.xlabel('cantidad de valores singulares')
        plt.ylabel('error de compresión en %')

        # set y axis limits
        plt.ylim(0.1, 100)

        plt.savefig(savename, format='pdf', dpi=1200)

    # get the mean of the values

    valores_totales = np.array(valores_totales)
    valores_totales = np.mean(valores_totales, axis=0)

    plt.plot(etiquetas, valores_totales, label='media', color='b', linestyle='solid')

    for i, valor in enumerate(valores_totales):
        print(f"con k = {i}", 'Error de compresión medio es: {:.2%}'.format(valor))

    plt.axhline(y=5, color='r', linestyle='--', label='5% de error')
    plt.legend()
    plt.savefig(savename, format='pdf', dpi=1200)
    plt.close()

if __name__ == '__main__':

    # file_path = "dataset_imagenes\img02.jpeg"

    # img_original = Image.open(file_path)

    # # compress the image with k = 4
    # mat_comprimida = compress_image(file_path, 3)

    # # primera figura
    # compare_compresion_save(mat_comprimida, img_original, 'comparacion_imagenes_4_k3.pdf')

    # # segunda figura
    # #plot singular values and save as pdf
    # U, S, VT = np.linalg.svd(img_original)
    # plt.plot(S)
    # # plt.semilogy()
    # plt.xlabel('índice')
    # plt.ylabel('valor singular')
    # plt.savefig('singular_values_1.pdf', format='pdf', dpi=1200)
    # plt.close()
    


    # # tercer figura
    # plot_compresion_lost_hist("dataset_imagenes\img02.jpeg", 28, 'error_de_compresion_hist_1.pdf')


    # cuarta figura
    # plot_compresion_lost_mean("dataset_imagenes/", 28, 'error_de_compresion_mean_1.pdf')

    # quinta figura
    plot_images_from_folder("dataset_imagenes/", 4, 0.4, 'imagenes_dataset.pdf')

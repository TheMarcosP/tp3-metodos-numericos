import pandas as pd
import matplotlib.pyplot as plt

def generar_imagen_csv(csv_file, output_file):
    # Leer el archivo CSV con pandas
    df = pd.read_csv(csv_file)

    # Crear la figura y los ejes de la tabla
    fig, ax = plt.subplots()

    # Eliminar los ejes
    ax.axis('off')

    # Crear la tabla a partir del dataframe
    tabla = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    # Establecer el formato de la tabla
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1.2, 1.2)

    # Guardar la figura como PDF
    plt.savefig(output_file, format='pdf', bbox_inches='tight')

    # Mostrar un mensaje de éxito
    print(f"Se ha generado la imagen en formato PDF: {output_file}")

# Llamar a la función con el nombre del archivo CSV y el nombre de salida del archivo PDF
generar_imagen_csv('tabla_dimensiones.csv', 'tabla.pdf')

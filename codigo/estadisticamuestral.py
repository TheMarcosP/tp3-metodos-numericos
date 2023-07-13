
def media(arreglo):
    return sum(arreglo)/len(arreglo)

def varianza(arreglo):
    m = media(arreglo)
    return sum([(x - m)**2 for x in arreglo])/(len(arreglo) - 1)

def desviacion_estandar(arreglo):
    return varianza(arreglo)**0.5

def covarianza(arreglo1, arreglo2):
    m1 = media(arreglo1)
    m2 = media(arreglo2)
    return sum([(x1 - m1)*(x2 - m2) for x1, x2 in zip(arreglo1, arreglo2)])/(len(arreglo1) - 1)


# #main

# arreglo = [3.2, 2.9, 1.8, 3.2, 1.9, 3.8, 2.5, 2.3, 2.7, 3.0, 3.7, 2.1, 2.4, 2.5, 2.8, 2.5, 2.8, 3.3, 2.0, 1.9]
# print("la media es: ", media(arreglo))
# print("la varianza es: ", varianza(arreglo))
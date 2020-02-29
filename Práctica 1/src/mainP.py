import numpy as np
from Neurona import *
from Enlace import *
from Neurona import *
from Enlace import *
from sklearn.model_selection import train_test_split
import sys

# Lectura del fichero

parametro1 = sys.argv[1]
parametro2 = sys.argv[2]

print(parametro2)

if type(parametro2) is str:

    # Modo 3
    print("Modo 3")
elif int(parametro2) == 100:

    # Modo 2
    print("Modo 2")
else:

    # Modo 1
    print("Modo 1")

    f = open(parametro1)
    lineas = f.readlines()

    # la primera linea
    primera_linea = lineas[0]

    # el resto
    for line in lineas[1:]:
        # TODO: hay que hacer la parte de coger X datos para train e Y para test
        # TODO: hay que ver como tratar los datos, si meterlos en lista o que
        pass

# Creando las neuronas

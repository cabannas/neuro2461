import numpy as np
from Neurona import *
from Enlace import *
import sys


# Lectura de fichero
f = open(sys.argv[1])
lineas_entrada = f.readlines()
f.close()
entradas = np.empty((0, 3), int)

for linea in lineas_entrada:
    entradas = np.concatenate((entradas, [list(map(int, linea.replace("\n", "").split(" ")))]))
entradas = np.concatenate((entradas, [[0, 0, 0], [0, 0, 0]]))

# Creando las Neuronas
capa0 = np.array([NeuronaMPX(), NeuronaMPX(), NeuronaMPX()])
capa1 = np.array([NeuronaMPZ(2), NeuronaMPZ(2), NeuronaMPZ(2)])
capa2 = np.array([NeuronaMPZ(2), NeuronaMPZ(2), NeuronaMPZ(2), NeuronaMPZ(2), NeuronaMPZ(2), NeuronaMPZ(2)])
capa3 = np.array([NeuronaMPY(2), NeuronaMPY(2)])

# Creando los Enlaces
for i, neurona in enumerate(capa0):
    enlace = Enlace(2, capa0[i], capa1[i])
    capa0[i].addEnlaceSalida(enlace)

enlace1 = Enlace(1, capa0[0], capa2[2])
enlace2 = Enlace(1, capa0[0], capa2[-1])
capa0[0].addEnlaceSalida(enlace1)
capa0[0].addEnlaceSalida(enlace2)
enlace1 = Enlace(1, capa0[1], capa2[1])
enlace2 = Enlace(1, capa0[1], capa2[-2])
capa0[1].addEnlaceSalida(enlace1)
capa0[1].addEnlaceSalida(enlace2)
enlace1 = Enlace(1, capa0[2], capa2[0])
enlace2 = Enlace(1, capa0[2], capa2[3])
capa0[2].addEnlaceSalida(enlace1)
capa0[2].addEnlaceSalida(enlace2)

for i, neurona in enumerate(capa1):
    enlace1 = Enlace(1, capa1[i], capa2[2 * i])
    enlace2 = Enlace(1, capa1[i], capa2[2 * i + 1])
    capa1[i].addEnlaceSalida(enlace1)
    capa1[i].addEnlaceSalida(enlace2)

for i, neurona in enumerate(capa2):
    if i % 2 == 0:
        enlace = Enlace(2, capa2[i], capa3[0])
        capa2[i].addEnlaceSalida(enlace)
    else:
        enlace = Enlace(2, capa2[i], capa3[1])
        capa2[i].addEnlaceSalida(enlace)

salida = np.array([])
capasIntermedias = np.array([capa1, capa2])

# Abrimos el fichero de escritura
f = open("../predicciones/McCulloch_Pitts.txt", "w")

# Ejecutamos la red Neuronal
for j, entrada in enumerate(entradas):
    for i, neurona in enumerate(capa0):
        neurona.recibirSenal(entrada[i])
    for capa in capasIntermedias:
        for neurona in capa:
            neurona.funcionActivacion()
    for k, neurona in enumerate(capa3):
        f.write(str(neurona.funcionActivacion()))
        if k < capa3.size - 1:
            f.write(" ")
    if j < entradas.shape[0] - 1:
        f.write("\n")

f.close()

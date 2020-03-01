##
import numpy as np
from Neurona import *
from Enlace import *
from sklearn.model_selection import train_test_split
import math
import sys
import random

# Lectura del fichero

if len(sys.argv) < 5:
    print("introduzca la cantidad de parámetros correcta, para mas información utilice el makefile")

modo = 0

if len(sys.argv) == 5:
    # Modo 2
    modo = 2
    print("Modo 2")
elif sys.argv[2].replace('.','',1).replace(',','',1).isdigit():
    # Modo 1
    modo = 1
    print("Modo 1")
else:
    # Modo 3
    modo = 3
    print("Modo 3")

# cosas que entrarán por argumento
if modo == 1:
    fichero = sys.argv[1]
    if sys.argv[2].isdigit():
        porcen_train = int(sys.argv[2])/100
    else:
        porcen_train = float(sys.argv[2])
    umbral = float(sys.argv[3])
    a = float(sys.argv[4])  # tasa de apredizaje
    maxEpocas = int(sys.argv[5])
elif modo == 2:
    fichero = sys.argv[1]
    umbral = float(sys.argv[2])
    a = float(sys.argv[3])  # tasa de apredizaje
    maxEpocas = int(sys.argv[4])
else:
    fichero = sys.argv[1]
    fichero2 = sys.argv[2]
    umbral = float(sys.argv[3])
    a = float(sys.argv[4])  # tasa de apredizaje
    maxEpocas = int(sys.argv[4])


# Lectura de fichero
f = open(fichero)
lineas_entrada = f.readlines()
f.close()

primera_linea = list(map(int, lineas_entrada[0].replace("\n", "").split(" ")))
atributos = primera_linea[0]
clases = primera_linea[1]

datos = np.empty((0, atributos + 1), float)

for linea in lineas_entrada[1:]:
    linea_cortada = list(map(float, ' '.join(linea.split()).replace("\n", "").replace("  "," ").split(" ")))
    dato = linea_cortada[0:atributos]
    # doy por hecho que siempre va a haber dos clases:
    if linea_cortada[-2] == 1:
        dato.append(1)
    else:
        dato.append(-1)

    datos = np.concatenate((datos, [dato]))


if modo == 3:
    # Lectura de fichero2
    f2 = open(fichero2)
    lineas_entrada2 = f2.readlines()
    f2.close()

    primera_linea2 = list(map(int, lineas_entrada2[0].replace("\n", "").split(" ")))
    atributos2 = primera_linea2[0]
    clases2 = primera_linea2[1]

    datos2 = np.empty((0, atributos2 + 1), float)

    for linea2 in lineas_entrada2[1:]:
        linea_cortada2 = list(map(float, linea2.replace("\n", "").split(" ")))
        dato2 = linea_cortada2[0:atributos2]

        if linea_cortada2[-2] == 1:
            dato2.append(1)
        else:
            dato2.append(-1)

        datos2 = np.concatenate((datos2, [dato2]))

# CREACION DE NEURONAS

capa0 = np.array([NeuronaPX() for p in range(atributos)])
capa1 = np.array([NeuronaPY(umbral)])

for neurona in capa0:
    enlace = Enlace(0, neurona, capa1[0])
    neurona.addEnlaceSalida(enlace)

# train y test
if modo == 1:  # en caso de que haya un porcentaje
    nDatosTrain = math.ceil(datos.shape[0] * porcen_train)
    indPerm = np.random.permutation(datos.shape[0])
    train = datos[indPerm[: nDatosTrain]]
    test = datos[indPerm[nDatosTrain:]]

elif modo == 2:  # en caso de que no sea con porcentaje y queramos coger todo
    train = datos
    test = datos

else:
    train = datos
    test = datos2

print("TRAIN =================")
print(train)
print("TEST =================")
print(test)

antiguob = 0
b = 0

pesos = []
for neurona in capa0:
    pesos.append(neurona.enlaceSalida.peso)

contadorEpocas = 0
while True:
    flag = False  # sin cambios en los pesos
    for entrada in train:
        for i, neurona in enumerate(capa0):
            neurona.recibirSenal(entrada[i])
        salida = capa1[0].funcionActivacion(b)
        if salida != entrada[-1]:
            for i, neurona in enumerate(capa0):
                neurona.enlaceSalida.cambiarPeso(neurona.enlaceSalida.peso + a * entrada[-1] * entrada[i])
            b = b + a * entrada[-1]
    for i, neurona in enumerate(capa0):
        if abs(neurona.enlaceSalida.peso - pesos[i]) != 0:
            flag = True
        pesos[i] = neurona.enlaceSalida.peso
    if abs(antiguob - b) != 0:
        flag = True
    antiguob = b
    # comparar bien los pesos (la bandera creo que no funciona o algo está mal porque no para)
    if not flag:
        print("Entrenamiento finalizado al no cambiar los pesos ni el sesgo.")
        break
    contadorEpocas += 1
    if contadorEpocas >= maxEpocas:
        print("Entrenamiento finalizado al alcanzar el número máximo de épocas.")
        break


print("Épocas realizadas:" + str (contadorEpocas))
print(pesos)
print(b)

# TEST
fallos = 0
for entrada in test:
    for i, neurona in enumerate(capa0):
        neurona.recibirSenal(entrada[i])
    salida = capa1[0].funcionActivacion(b)

    if salida != entrada[-1]:
        fallos += 1

tasaError = fallos / len(test) * 100

print("Tasa Error: " + str(tasaError) + " %\n")


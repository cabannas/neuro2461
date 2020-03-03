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
if sys.argv[2].isdigit() and int(sys.argv[2]) < 100:
    # Modo 1
    modo = 1
    print("Modo 1")
elif sys.argv[2].isdigit() and int(sys.argv[2]) == 100:
    # Modo 2
    modo = 2
    print("Modo 2")
else:
    # Modo 3
    modo = 3
    print("Modo 3")

porcen_train = 0
fichero2 = "none"
# cosas que entrarán por argumento

fichero = sys.argv[1]

if modo == 3:
    fichero2 = sys.argv[2]

else:  # Modo 1 y Modo 2

    porcen_train = int(sys.argv[2]) / 100

umbral = float(sys.argv[3])
a = float(sys.argv[4])  # tasa de apredizaje
maxEpocas = int(sys.argv[5])

# Lectura de fichero
f = open(fichero)
lineas_entrada = f.readlines()
f.close()

primera_linea = list(map(int, lineas_entrada[0].replace("\n", "").split(" ")))
atributos = primera_linea[0]
clases = primera_linea[1]

datos = np.empty((0, atributos + 1), float)

for linea in lineas_entrada[1:]:
    linea_cortada = list(map(float, ' '.join(linea.split()).replace("\n", "").replace("  ", " ").split(" ")))
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

    # Abriendo fichero preficciones
    fp = open("../predicciones/Perceptron.txt", "w")

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


antiguob = 0
b = 0

# AJUSTE DE PESOS

pesos = []
for neurona in capa0:
    pesos.append(neurona.enlaceSalida.peso)

# ENTRENAMIENTO DE LA RED

listaECMsTrain = []
contadorEpocas = 0
VP = 0
VN = 0
FP = 0
FN = 0
while True:
    errorCuadraticoTrain = 0
    fallosTrain = 0
    flag = False  # sin cambios en los pesos
    for entrada in train:
        for i, neurona in enumerate(capa0):
            neurona.recibirSenal(entrada[i])
        salida = capa1[0].funcionActivacion(b)
        if salida != entrada[-1]:
            fallosTrain += 1
            for i, neurona in enumerate(capa0):
                neurona.enlaceSalida.cambiarPeso(neurona.enlaceSalida.peso + a * entrada[-1] * entrada[i])
            b = b + a * entrada[-1]
            if salida  == 1:
                FP += 1
            else:
                FN += 1
        else:
            if salida == 1:
                VP += 1
            else:
                VN += 1            
        errorCuadraticoTrain += (entrada[-1] - salida) ** 2
    for i, neurona in enumerate(capa0):
        if abs(neurona.enlaceSalida.peso - pesos[i]) != 0:
            flag = True
        pesos[i] = neurona.enlaceSalida.peso
    if abs(antiguob - b) != 0:
        flag = True
    antiguob = b

    contadorEpocas += 1
    if not flag:
        print("\nEntrenamiento finalizado al no cambiar los pesos ni el sesgo.")
        break
    if contadorEpocas >= maxEpocas:
        print("\nEntrenamiento finalizado al alcanzar el número máximo de épocas.")
        break

tasaErrorTrain = fallosTrain / len(train) * 100
errorCuadraticoMedioTrain = errorCuadraticoTrain / len(train)
listaECMsTrain.append(errorCuadraticoMedioTrain)

print("\nÉpocas realizadas:" + str(contadorEpocas))
print("\nTasa Error en Train: " + str(tasaErrorTrain) + " %")
print("Error cuadrático en Train: " + str(errorCuadraticoMedioTrain))

print("Matriz de confusión en Train:")

titlesX = ['', 'Valor real = 1', 'Valor real = -1']
titlesY = ['Valor estimado = 1', 'Valor estimado = -1']
data = [titlesX] + list(zip(titlesY, [VP,FP], [FN,VN]))

for i, d in enumerate(data):
    line = '|'.join(str(x).ljust(len("Valor estimado = -1")) for x in d)
    print(line)
    if i == 0:
        print('-' * (len(line)+len("Valor estimado = -1")))


# TESTEO DE LA RED

errorCuadraticoTest = 0
fallosTest = 0
VP = 0
VN = 0
FP = 0
FN = 0
if modo ==3:
    fp.write(str(atributos) + " " + str(clases) + " \n")
for entrada in test:
    for i, neurona in enumerate(capa0):
        neurona.recibirSenal(entrada[i])
    salida = capa1[0].funcionActivacion(b)
    if modo == 3:
        for e in entrada[:-1]:
            if(e==0):
                fp.write("0")
            else:
                fp.write(str(e))
            fp.write(" ")
        if salida == 1:
            fp.write('1 ')
        else:
            fp.write('0 ')
        if salida == -1:
            fp.write('1\n')
        else:
            fp.write('0\n')

    else:
        errorCuadraticoTest += (entrada[-1] - salida)**2
        if salida != entrada[-1]:
            fallosTest += 1
            if salida  == 1:
                FP += 1
            else:
                FN += 1
        else:
            if salida == 1:
                VP += 1
            else:
                VN += 1

if modo == 3:
    fp.close()
else:
    tasaErrorTest = fallosTest / len(test) * 100
    errorCuadraticoMedioTest = errorCuadraticoTest / len(test)

    print("\nTasa Error en Test: " + str(tasaErrorTest) + " %")
    print("Error cuadrático en Test: " + str(errorCuadraticoMedioTest))
    print("Matriz de confusión en Test:")
    titlesX = ['', 'Valor real = 1', 'Valor real = -1']
    titlesY = ['Valor estimado = 1', 'Valor estimado = -1']
    data = [titlesX] + list(zip(titlesY, [VP,FP], [FN,VN]))

    for i, d in enumerate(data):
        line = '|'.join(str(x).ljust(len("Valor estimado = -1")) for x in d)
        print(line)
        if i == 0:
            print('-' * (len(line)+len("Valor estimado = -1")))
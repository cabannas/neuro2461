##
import numpy as np
from Neurona import *
from Enlace import *
from sklearn.model_selection import train_test_split

# cosas que entrarán por argumento
fichero = "problema_real1.txt"
umbral = 0.2
porcen_train = 0.7
a = 0.3  # tasa de aprendizaje
maxEpocas = 100

# Lectura de fichero
f = open(fichero)
lineas_entrada = f.readlines()
f.close()

primera_linea = list(map(int, lineas_entrada[0].replace("\n", "").split(" ")))
parametros = primera_linea[0]
clases = primera_linea[1]
print(clases)

datos = np.empty((0, parametros + clases), float)

# for i in range lineas_entrada[0]

for linea in lineas_entrada[1:]:
    linea_cortada = list(map(float, linea.replace("\n", "").split(" ")))
    dato = linea_cortada[0:parametros]
    # doy por hecho que siempre va a haber dos clases:
    if linea_cortada[-2] == 1:
        dato.append(1)
    else:
        dato.append(-1)

    if linea_cortada[-1] == 1:
        dato.append(1)
    else:
        dato.append(-1)

    datos = np.concatenate((datos, [dato]))
print(datos)

capa0 = np.array([NeuronaPX() for p in range(parametros)])
print(capa0)
capa1 = np.array([NeuronaPY(umbral)])

for neurona in capa0:
    enlace = Enlace(0, neurona, capa1[0])
    neurona.addEnlaceSalida(enlace)

# TODO: hay que cambiar esto por lo de FAA
train, test = train_test_split(datos, test_size=1 - porcen_train)
b = 0

pesos = []
for neurona in capa0:
    pesos.append(neurona.enlaceSalida.peso)

contadorEpocas = 0
while True and contadorEpocas < maxEpocas:
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
        if neurona.enlaceSalida.peso - pesos[i]:
            flag = True
            pesos[i] = neurona.enlaceSalida.peso
    print(b)
    # comparar bien los pesos (la bandera creo que no funciona o algo está mal porque no para)
    if not flag:
        break
    contadorEpocas += 1

##
import numpy as np
from Neurona import *
from Enlace import *
from sklearn.model_selection import train_test_split

# cosas que entrarán por argumento
fichero = "problema_real1.txt"
umbral = 0.2
porcen_train = 0.7
maxEpocas = 100
a = 0.1  # TASA DE APRENDIZAJE

# Lectura de fichero
f = open(fichero)
lineas_entrada = f.readlines()
f.close()

primera_linea = list(map(int, lineas_entrada[0].replace("\n", "").split(" ")))
entradas = primera_linea[0]
salidas = primera_linea[1]

datos = np.empty((0, entradas + salidas), float)

# for i in range lineas_entrada[0]

for linea in lineas_entrada[1:]:
    linea_cortada = list(map(float, linea.replace("\n", "").split(" ")))
    dato = linea_cortada[0:entradas]
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

# Creamos las neuronas

capa0 = np.array([NeuronaPX() for p in range(entradas)])
capa1 = np.array([NeuronaPY(umbral) for p in range(salidas)])

# Inicializamos todos los pesos y el sesgo a 0
for neurona in capa0:
    enlace = Enlace(0, neurona, capa1[0])
    neurona.addEnlaceSalida(enlace)

for neurona in capa0:
    enlace = Enlace(0, neurona, capa1[1])
    neurona.addEnlaceSalida(enlace)

train, test = train_test_split(datos, test_size=1 - porcen_train)


# Bias, uno por cada salida
b1 = 0
b2 = 0

pesosSalida1 = []
pesosSalida2 = []

for neurona in capa0:
    pesosSalida1.append(neurona.enlacesSalida[0])

for neurona in capa0:
    pesosSalida2.append(neurona.enlacesSalida[1])

condicionParada = False
contadorEpocas = 0

while not condicionParada:

    ocurreCambio = False

    for valorEntrada in train:
        # Neuronas de la capa de entrada
        for i, neurona in enumerate(capa0):
            neurona.recibirSenal(valorEntrada[i])
        # Neuronas de la capa de salida
        salida1 = capa1[0].funcionActivacion(b1)
        salida2 = capa1[1].funcionActivacion(b2)
        if salida1 != valorEntrada[-2]:
            for i, neurona in enumerate(capa0):
                neurona.enlacesSalida[0].cambiarPeso(neurona.enlacesSalida[0].peso + a * valorEntrada[-2] * valorEntrada[i])
            b1 = b1 + a * valorEntrada[-2]
            ocurreCambio = True
        if salida2 != valorEntrada[-1]:
            for i, neurona in enumerate(capa0):
                neurona.enlacesSalida[1].cambiarPeso(neurona.enlacesSalida[1].peso + a * valorEntrada[-1] * valorEntrada[i])
            b2 = b2 + a * valorEntrada[-1]
            ocurreCambio = True
    contadorEpocas += 1
    if not ocurreCambio or contadorEpocas == maxEpocas:
        condicionParada = True

    print("bias 1 = " + str(b1))
    print("bias 2 = " + str(b2))
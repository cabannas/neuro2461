import numpy as np
from abc import ABCMeta, abstractmethod

from Enlace import Enlace


class Neurona:
    # Clase abstracta
    __metaclass__ = ABCMeta

    @abstractmethod
    def funcionActivacion(self):
        pass

    @abstractmethod
    def recibirSenal(self, entrada):
        pass

    # def addEntrada(self, enlace):
    #    self.enlacesEntrada = np.append(self.enlacesEntrada, enlace)

    # def addSalida(self, enlace):
    #   self.enlacesSalida = np.append(self.enlacesSalida, enlace)


class NeuronaMPZ(Neurona):
    def __init__(self, umbral):
        self.umbral = umbral
        self.enlacesSalida = np.array([])
        self.enlacesEntrada = np.array([])
        self.valores = np.array([])
        self.valor = 0

    def recibirSenal(self, entrada):
        self.valores = np.append(self.valores, entrada)

    def funcionActivacion(self):
        if np.sum(self.valores) < self.umbral:
            nuevo_valor = 0
        else:
            nuevo_valor = 1
        self.valores = np.array([])
        valor_a_enviar = self.valor
        self.valor = nuevo_valor
        for enlace in self.enlacesSalida:
            enlace.enviarValor(valor_a_enviar)

    def addEnlaceSalida(self, enlace):
        self.enlacesSalida = np.append(self.enlacesSalida, enlace)


class NeuronaMPX(Neurona):

    def __init__(self):
        self.enlacesSalida = np.array([])
        self.valor = 0

    def recibirSenal(self, entrada):
        for enlace in self.enlacesSalida:
            enlace.enviarValor(entrada)

    def funcionActivacion(self):
        pass

    def addEnlaceSalida(self, enlace):
        self.enlacesSalida = np.append(self.enlacesSalida, enlace)


class NeuronaMPY(Neurona):

    def __init__(self, umbral):
        self.umbral = umbral
        self.valores = np.array([])
        self.valor = 0

    def recibirSenal(self, entrada):
        self.valores = np.append(self.valores, entrada)

    def funcionActivacion(self):
        if np.sum(self.valores) < self.umbral:
            nuevo_valor = 0
        else:
            nuevo_valor = 1
        self.valores = np.array([])
        valor_a_enviar = self.valor
        self.valor = nuevo_valor
        return valor_a_enviar


class NeuronaPX(Neurona):

    def __init__(self):
        self.enlaceSalida = Enlace()  # No se si esto está bien, no recuerdo como lo hiciste tu

    def recibirSenal(self, entrada):
        self.enlaceSalida.enviarValor(entrada)

    def funcionActivacion(self):
        pass

    def addEnlaceSalida(self, enlace):
        self.enlaceSalida = enlace


class NeuronaPY(Neurona):

    def __init__(self, umbral):
        self.umbral = umbral
        self.valores = np.array([])

    def recibirSenal(self, entrada):
        self.valores = np.append(self.valores, entrada)

    def funcionActivacion(self, b):  # Esto puede dar problemas, ya que no sigue la estructura de la clase Padre
        if np.sum(self.valores) + b > self.umbral:
            salida = 1
        elif np.sum(self.valores) + b < -self.umbral:
            salida = -1
        else:
            salida = 0
        self.valores = np.array([])
        return salida

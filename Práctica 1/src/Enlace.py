class Enlace:
    def __init__(self, peso, neuronaEntrada, neuronaSalida):
        self.peso = peso
        self.neuronaEntrada = neuronaEntrada
        self.neuronaSalida = neuronaSalida
        
    def enviarValor(self, valor):
        self.neuronaSalida.recibirSenal(valor*self.peso)

    def cambiarPeso(self, peso):
        self.peso = peso

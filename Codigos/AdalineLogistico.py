import pandas as pd
import numpy as np

class AdalineLogistico:
    """
    Implementação do neurônio ADALINE com função de ativação sigmoide (Regressão Logística)
    usando o Gradiente Descendente.
    """
    def __init__(self, taxaAprendizado=0.01, nEpocas=100, estagioAleatorio=42):
        self.taxaAprendizado = taxaAprendizado
        self.nEpocas = nEpocas
        self.estagioAleatorio = estagioAleatorio
        self.pesos = None
        self.bias = None
        self.custos = []
        
    def inicializarParametros(self, nCaracteristicas):
        """ Inicializa os pesos com pequenos valores aleatórios e o bias como zero. """
        geradorAleatorio = np.random.RandomState(self.estagioAleatorio)
        self.pesos = geradorAleatorio.normal(loc=0.0, scale=0.01, size=nCaracteristicas)
        self.bias = 0.0
        
    
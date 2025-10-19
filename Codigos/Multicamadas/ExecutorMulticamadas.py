# Biblioteca usada somente para separar os dados em treino e teste.
from sklearn.model_selection import train_test_split
from Multicamadas.Multicamadas import Multicamadas
from AdalineLogistica.GraficosAdalineLogistica import GraficosAdalineLogistica
import numpy as np

class ExecutorMulticamadas:
    """
    Classe dedicada a executar o pipeline completo de treinamento, avaliação
    e visualização para o modelo de Redes Neurais de Múltiplas Camadas.
    """
    
    """Construtor recebe os conjuntos de dados já divididos"""
    def __init__(self, xTreinoVal, yTreinoVal, xTeste, yTeste):
            self.xTreinoVal = xTreinoVal
            self.yTreinoVal = yTreinoVal
            self.xTeste = xTeste
            self.yTeste = yTeste
            
    """Método que orquestra todo o processo para modelo Multicamadas."""
    def rodar(self):
        print("Iniciando o treinamento do modelo Multicamadas...")
        #Divide os dados de treino em subconjuntos de treino e validação
        xTreino, xVal, yTreino, yVal = train_test_split(
            self.xTreinoVal, self.yTreinoVal,
            test_size=0.2,
            random_state=42,
            stratify=self.yTreinoVal
        )
        # Define a arquitetura da rede neural
        nEntrada = xTreino.shape[1]  # Número de características
        nOculta = 10                 # Número de neurônios na camada oculta (hiperparâmetro)
        nSaida = 1                   # Número de neurônios na camada de saída (1 para classificação binária)
        # Instancia e treina o modelo
        mlp = Multicamadas(
            nEntrada=nEntrada,
            nOculta=nOculta,
            nSaida=nSaida,
            taxaAprendizado=0.0001,
            nEpocas=30000
        )
        mlp.fit(xTreino.values, yTreino.values, xVal.values, yVal.values)
        # Avalia o modelo no conjunto de teste final
        yPred = mlp.predict(self.xTeste.values)
        acuracia = mlp.acuracia(self.xTeste.values, self.yTeste.values) * 100
        print(f"Acurácia do modelo Multicamadas: {acuracia:.2f}%")
        # Gera e exibe os gráficos de análise
        print("Gerando gráficos de análise para o modelo Multicamadas...")
        GraficosAdalineLogistica.plotarCurvaErro(mlp)
        GraficosAdalineLogistica.plotarMatrizConfusao(self.yTeste.values, yPred, classes=['Não Doente', 'Doente'])
        GraficosAdalineLogistica.plotarCurvasAprendizado(mlp)
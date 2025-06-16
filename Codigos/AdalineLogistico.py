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
        
    def sigmoid(self, z):
        """ Função sigmoide. Pega um valor real e o espreme entre 0 e 1, 
        para gerar uma probabilidade. """
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))
    
    def fit(self, X, y):
        """ Ajusta os pesos e o bias (desvio) aos dados de treinamento. """
        nFaetures = X.shape[1]
        self.inicializarParametros(nFaetures)
        
        for i in range(self.nEpocas):
            # Calcula a entrada líquida e a ativação (probabilidade)
            entradaLiquida = np.dot(X, self.pesos) + self.bias
            ativacao = self.sigmoid(entradaLiquida)
            # Calcula o erro
            erro = y - ativacao
            # Calula os gradientes (derivadas)
            gradientesPesos = -X.T.dot(erro)
            gradientesBias = -np.sum(erro)
            # Atualiza os pesos e o bias
            self.pesos -= self.taxaAprendizado * gradientesPesos
            self.bias -= self.taxaAprendizado * gradientesBias
            # Calcula e armazena o custo (log loss)
            custo = -np.sum(y * np.log(ativacao) + (1 - y) * np.log(1 - ativacao))
            self.custos.append(custo)
        return self
    
    def predicProba(self, X):
        """ Retorna as probabilidades previstas para cada amostra. """
        entradaLiquida = np.dot(X, self.pesos) + self.bias
        return self.sigmoid(entradaLiquida)
    
    def predict(self, X, limiar=0.5):
        """ Faz a classificação final (0 ou 1) com base em um limiar. """
        return np.where(self.predictProba(X) >= limiar, 1, 0)
    
import numpy as np

class Multicamadas:
    """
    Construtor da classe.
    Parâmetros:
    - n_entrada: Número de neurônios na camada de entrada (igual ao número de características do dataset).
    - n_oculta: Número de neurônios na camada oculta.
    - n_saida: Número de neurônios na camada de saída (1 para classificação binária).
    """
    def __init__(self, nEntrada, nOculta, nSaida, taxaAprendizado=0.001, nEpocas=100, estagioAleatorio=42):
        self.nEntrada = nEntrada
        self.nOculta = nOculta
        self.nSaida = nSaida
        self.taxaAprendizado = taxaAprendizado
        self.nEpocas = nEpocas
        self.estagioAleatorio = estagioAleatorio
        # Atributos para análise, compatíveis com a classe de visualização
        self.erros = []
        self.historicoAcuraciaTreino = []
        self.historicoAcuraciaVal = []
        # Inicialização dos pesos e bias
        self.pesosOculta, self.biasOculta, self.pesosSaida, self.biasSaida = self.inicializarParametros()
        
    """Inicializa os pesos e bias para todas as camadas com valores aleatórios pequenos."""
    def inicializarParametros(self):
        geradorAleatorio = np.random.RandomState(self.estagioAleatorio)
        # Pesos e bias entre a camada de entrada e a camada oculta
        limiteOculta = np.sqrt(6 / (self.nEntrada + self.nOculta))
        pesosOculta = geradorAleatorio.uniform(-limiteOculta, limiteOculta, (self.nEntrada, self.nOculta))
        biasOculta = np.zeros((1, self.nOculta))
        # Pesos e bias entre a camada oculta e a camada de saída
        limiteSaida = np.sqrt(6 / (self.nOculta + self.nSaida))
        pesosSaida = geradorAleatorio.uniform(-limiteSaida, limiteSaida, (self.nOculta, self.nSaida))
        biasSaida = np.zeros((1, self.nSaida))
        return pesosOculta, biasOculta, pesosSaida, biasSaida
    
    # Ativação ReLU na camada oculta
    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivada(self, z):
        return (z > 0).astype(float)
    
    """Função de ativação sigmoide."""
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))
    
    """Derivada da função sigmoide, necessária para o backpropagation."""
    def sigmoid_derivada(self, ativacaoSigmoid):
        return ativacaoSigmoid * (1 - ativacaoSigmoid)
    
    """Treina a rede neural usando o algoritmo de backpropagation."""
    def fit (self, X, y, xVal=None, yVal=None):
        # Garantir que y seja uma matriz coluna
        y = y.reshape(-1, 1)
        for i in range(self.nEpocas):
            # --- 1. FORWARD PROPAGATION ---
            # Camada de entrada para camada oculta
            entradaOculta = np.dot(X, self.pesosOculta) + self.biasOculta
            ativacaoOculta = self.relu(entradaOculta)
            # Camada oculta para camada de saída
            entradaSaida = np.dot(ativacaoOculta, self.pesosSaida) + self.biasSaida
            ativacaoSaida = self.sigmoid(entradaSaida)
            # --- 2. CÁLCULO DO ERRO ---
            erroVetor = ativacaoSaida - y
            erroQuadraticoMedio = np.mean(erroVetor ** 2)
            self.erros.append(erroQuadraticoMedio)
            # --- 3. BACKPROPAGATION ---
            # Gradiente da camada de saída
            dSaida = erroVetor * self.sigmoid_derivada(ativacaoSaida)
            # Gradiente da camada oculta
            dOculta = np.dot(dSaida, self.pesosSaida.T) * self.relu_derivada(entradaOculta)
            # --- 4. ATUALIZAÇÃO DOS PESOS E BIAS ---
            # Atualiza pesos e bias da camada de saída
            self.pesosSaida -= self.taxaAprendizado * np.dot(ativacaoOculta.T, dSaida)
            self.biasSaida -= self.taxaAprendizado * np.sum(dSaida, axis=0, keepdims=True)
            # Atualiza pesos e bias da camada oculta
            self.pesosOculta -= self.taxaAprendizado * np.dot(X.T, dOculta)
            self.biasOculta -= self.taxaAprendizado * np.sum(dOculta, axis=0, keepdims=True)
            # Salva histórico de acurácia para os gráficos
            self.historicoAcuraciaTreino.append(self.acuracia(X, y.flatten()))
            if xVal is not None and yVal is not None:
                self.historicoAcuraciaVal.append(self.acuracia(xVal, yVal))
        return self
    
    """Faz a predição de probabilidades"""
    def predictProba(self, X):
        entradaOculta = np.dot(X, self.pesosOculta) + self.biasOculta
        ativacaoOculta = self.relu(entradaOculta)
        entradaSaida = np.dot(ativacaoOculta, self.pesosSaida) + self.biasSaida
        return self.sigmoid(entradaSaida).flatten()
    
    """Faz a predição de classes (0 ou 1)"""
    def predict(self, X, limiar=0.5):
        return np.where(self.predictProba(X) >= limiar, 1, 0)
    
    """Calcula a acurácia do modelo."""
    def acuracia(self, X, yReal):
        previsoes = self.predict(X)
        return np.mean(previsoes == yReal)

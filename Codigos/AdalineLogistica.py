import numpy as np

class AdalineLogistica:
    """
    Implementação do neurônio ADALINE com função de ativação sigmoide (Regressão Logística)
    usando o Gradiente Descendente.
    """
    
    #Construtor da classe que configura os iniciais parâmetros do modelo
    def __init__(self, taxaAprendizado=0.01, nEpocas=100, estagioAleatorio=42):
        #taxaAprendizado: Valor que representa o tamanho do passo que vai ser dado a cada aprendizado.
        #valor de 0.01 é um valor intermediario bom para não dar passos muito grandes e nem muito curtos.
        self.taxaAprendizado = taxaAprendizado
        #nEpocas: Número de passagens completas por todo o conjunto de dados.
        #valor de 100 é um valor intermediario bom para não dar muitas passagens e nem poucas.
        self.nEpocas = nEpocas
        #estagioAleatorio: Gera sempre os mesmos números aleatórios, para que o modelo seja reprodutível.
        #valor de 42 é um valor comum para garantir reprodutibilidade.
        self.estagioAleatorio = estagioAleatorio
        #pesos: Valor que representa a importância de cada característica na previsão, como idade, colesterol, etc.
        self.pesos = None
        #bias: Valor que representa o deslocamento da função de ativação, ajustando a linha de decisão.
        self.bias = None
        #custos: Lista que armazena o custo a cada época, para monitorar o aprendizado, se o custo for diminuindo é 
        #a prova de que o modelo está aprendendo.
        self.custos = []
    
    #Inicializa os pesos e o bias, dando um ponto de partida para o aprendizado.    
    def inicializarParametros(self, nCaracteristicas):
        #Gerador de números aleatórios baseados no estagioAleatorio para manter a reprodutibilidade.
        geradorAleatorio = np.random.RandomState(self.estagioAleatorio)
        #Cria um vetor de pesos com valores aleatórios próximos de zero para cada caracteristica.
        #Uma curiosidade, podemos indicar qual seria esse valor inicial para cada característica de forma coerente, 
        #fazendo isso pode acelerar o aprendizado, porém o modelo garante que vai achado o peso ideal para cada característica, então passar esse valor inicial para elas pode induzir a ter um resultado imparcial, atrapalhando o resultado.
        self.pesos = geradorAleatorio.normal(loc=0.0, scale=0.01, size=nCaracteristicas)
        #Inicializa o bias com zero, por ser uma prática padrão, segura e eficiente.
        self.bias = 0.0
    
    #Implementa a função sigmoide, que transforma a saída linear do modelo em uma probabilidade, um valor entre 0 e 1.
    def sigmoid(self, z):
        #Formula sigmoide, com np.clip para evitar overflow e underflow numérico, garantindo que os valores não sejam muito grandes ou muito pequenos.
        #e com np.exp para calcular a exponencial de z, que é a base da função sigmoide.
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
    
    def predictProba(self, X):
        """ Retorna as probabilidades previstas para cada amostra. """
        entradaLiquida = np.dot(X, self.pesos) + self.bias
        return self.sigmoid(entradaLiquida)
    
    def predict(self, X, limiar=0.5):
        """ Faz a classificação final (0 ou 1) com base em um limiar. """
        return np.where(self.predictProba(X) >= limiar, 1, 0)
    
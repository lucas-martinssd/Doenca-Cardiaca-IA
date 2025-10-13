#Biblioteca para realizart calculos matemáticos.
import numpy as np

class AdalineLogistica:
    """ É a implementação de um único neurônio que usa a Regressão Logística como sua base matemática. 
    Ele é adaptativo porque aprende de forma iterativa usando o Gradiente Descendente para ajustar seus parâmetros. """
    
    """ __init__: Construtor da classe que configura os parâmetros iniciais do modelo """
    def __init__(self, taxaAprendizado=0.0001, nEpocas=100, estagioAleatorio=42):
        #taxaAprendizado: Valor que representa o tamanho do passo que vai ser dado a cada aprendizado.
        #valor de 0.001 é um valor intermediario bom para não dar passos muito grandes e nem muito curtos.
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
        #Atributos para as curvas de aprendizado
        self.historicoAcuraciaTreino = []
        self.historicoAcuraciaVal = []
    
    """ inicializarParametros: Inicializa os pesos e o bias, dando um ponto de partida para o aprendizado."""
    def inicializarParametros(self, nCaracteristicas):
        #Gerador de números aleatórios baseados no estagioAleatorio para manter a reprodutibilidade.
        geradorAleatorio = np.random.RandomState(self.estagioAleatorio)
        #Cria um vetor de pesos com valores aleatórios próximos de zero para cada caracteristica.
        #Uma curiosidade, podemos indicar qual seria esse valor inicial para cada característica de forma coerente, 
        #fazendo isso pode acelerar o aprendizado, porém o modelo garante que vai achado o peso ideal para cada característica, 
        #então passar esse valor inicial para elas pode induzir a ter um resultado imparcial, atrapalhando o resultado.
        self.pesos = geradorAleatorio.normal(loc=0.0, scale=0.01, size=nCaracteristicas)
        #Inicializa o bias com zero, por ser uma prática padrão, segura e eficiente.
        self.bias = 0.0
    
    """ sigmoid: Implementa a função sigmoide, que transforma a saída linear do modelo em uma probabilidade, um valor entre 0 e 1."""
    def sigmoid(self, z):
        #Formula sigmoide, com np.clip para evitar overflow e underflow numérico, garantindo que os valores não sejam muito grandes ou 
        # muito pequenos.
        #e com np.exp para calcular a exponencial de z, que é a base da função sigmoide.
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))
    
    """ acuracia: Calcula a acurácia do modelo, que é a proporção de previsões corretas em relação ao total de previsões feitas."""
    def acuracia(self, yReal, X):
        previsoes = self.predict(X)
        return np.mean(yReal == previsoes)
    
    """ fit: Método que implementa o algoritmo de aprendizado do modelo Gradiente Descendente, ajustando os pesos e o bias com base nos 
        dados de treinamento. É o coração do modelo, onde o aprendizado realmente acontece. Método de treinamento."""
    def fit(self, X, y, XVal=None, yVal=None):
        #Conta o número de características (colunas) em X.
        nFaetures = X.shape[1]
        #Chama o método para inicializar os parâmetros do modelo.
        self.inicializarParametros(nFaetures)
        
        #Inicializa o loop de treinamento que repetirá pelo número de épocas definido.
        for i in range(self.nEpocas):
            #Calcula a soma ponderada das entradas (X) com os pesos e adiciona o bias.
            #Isso gera a entrada líquida do modelo, que é a base para calcular a ativação.
            #np.dot é usado para multiplicar a matriz X pelos pesos.
            entradaLiquida = np.dot(X, self.pesos) + self.bias
            #Pega a entradaLiquida e passa pela função sigmoide para obter as probabilidades previstas.
            ativacao = self.sigmoid(entradaLiquida)
            #Calcula o erro, fazendo a diferença entre as saídas reais (y, coluna Doença Cardíaca) e as previstas (ativacao).
            erro = y - ativacao
            #Calcula o gradiante dos pesos, ou seja, calcula a culpa de cada peso no erro total.
            gradientesPesos = -X.T.dot(erro)
            #Calcula o gradiente do bias, ele mede a tendência geral de erro do modelo.
            gradientesBias = -np.sum(erro)
            #Linha principal do loop, aonde o aprendizado acontece, pois é quando o peso de cada característica é alterado de acordo
            #com os aprendizados feitos até agora.
            self.pesos -= self.taxaAprendizado * gradientesPesos
            #Atualiza o bias com a mesma lógica dos pesos.
            self.bias -= self.taxaAprendizado * gradientesBias
            #Calcula o custo total para está época, medindo o quão errado o modelo está.
            custo = -np.sum(y * np.log(ativacao) + (1 - y) * np.log(1 - ativacao))
            #Armazena o custo na lista, para permitir a visualização do progresso.
            self.custos.append(custo)
            #Lógica para salvar a acurácia a cada época
            self.historicoAcuraciaTreino.append(self.acuracia(y, X))
            if XVal is not None and yVal is not None:
                self.historicoAcuraciaVal.append(self.acuracia(yVal, XVal))
        #Retorna o próprio objeto para permitir encadeamento de métodos.
        return self
    
    """ predictProba: Método que retorna as probabilidades previstas para cada paciente. """
    def predictProba(self, X):
        #Repete o cálculo linear, mas agora com os pesos finais otimizados.
        entradaLiquida = np.dot(X, self.pesos) + self.bias
        #Retorna a probabilidade final.
        return self.sigmoid(entradaLiquida)
    
    """ predict: Faz a classificação final (0 ou 1) com base em um limiar. Se for maior ou igual a 0.5, classifica como 1 
        (doença cardíaca presente),"""
    def predict(self, X, limiar=0.5):
        #Retorna 1 se a probabilidade prevista for maior ou igual ao limiar, caso contrário retorna 0.
        return np.where(self.predictProba(X) >= limiar, 1, 0)
    
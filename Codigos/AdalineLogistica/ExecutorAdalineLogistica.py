import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from AdalineLogistica.AdalineLogistica import AdalineLogistica
from Graficos.GraficosAdalineLogistica import GraficosAdalineLogistica

def meuTrainTestSplit(X, y, testeSize=0.3, randomState=42):
    """ Tem a função de dividir o conjunto de dados em dois grupos, um para treinar o modelo e o
        outro completamente separado, para testá-lo. """
    #Linhas que garantem que os índices estejam limpos e de forma sequencial como (1, 2, 3).
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    #Cria um array de números de 0 até o total de pacientes, para representar cada um.
    indices = np.arange(len(y))
    #Vai pegar todo o array de cada paciente e retornar apenas [0, 1].
    classes = np.unique(y)
    #Arrays que vão armazenar os índices dos pacientes de teste e treino.
    indicesTreino, indicesTeste = [], []
    #Lógica que vai ser usada para o embaralhamento de forma controlada usando o 42.
    rgen = np.random.RandomState(randomState)
    """ Loop que vai realizar a divisão dos pacientes de forma igual, ou seja, contendo o mesmo número de pacientes saudáveis e doentes,
        para treino e para teste. O loop vai ocorrer uma vez para a classe 0 saudáveis e uma para a classe 1 doentes. """
    for cls in classes:
        #Essa linha vai selecionar os índices dos pacientes que pertencem a classe do loop atual, 0 saudável ou 1 doente.
        indicesClasse = indices[y == cls]
        #Embaralha aleatoriamente os índices dentro daquela classe do loop.
        rgen.shuffle(indicesClasse)
        #Calcula a porcentagem de pacientes que vão ser usados para o treino com base no testeSize que já está definido 30%,
        # ou seja o treino é 70%;
        pontoCorte = int(len(indicesClasse) * (1 - testeSize))
        #Divide a lista de índices embaralhados da classe atual. indicesTreino tem 70% dos pacientes e indicesTeste tem 30%.
        indicesTreino.extend(indicesClasse[:pontoCorte])
        indicesTeste.extend(indicesClasse[pontoCorte:])
    #Realiza o embaralhamento controlado dos pacientes dentro do treino e do teste.
    rgen.shuffle(indicesTreino)
    rgen.shuffle(indicesTeste)
    #Pega os arrays já prontos e embaralhados e coloca cada um em seu respectivo lugar do banco X ou y.
    XTrain = X.iloc[indicesTreino]
    XTest = X.iloc[indicesTeste]
    yTrain = y.iloc[indicesTreino]
    yTest = y.iloc[indicesTeste]
    #Retorna os quatro conjuntos de dados.
    return XTrain, XTest, yTrain, yTest

class ExecutorAdalineLogistica:
    """
    Orquestra o carregamento, pré-processamento, treinamento, avaliação
    e visualização do modelo AdalineLogistica (NumPy).
    """
    def __init__(self, caminho_arquivo, random_state=42):
        """
        Construtor da classe.
        Parâmetros:
        - caminho_arquivo: String com o caminho para o arquivo CSV do banco de dados completo.
        - random_state: Semente para reprodutibilidade nas divisões de dados.
        """
        self.caminho_arquivo = caminho_arquivo
        self.random_state = random_state
        self.modelo = None
        # Atributos para os dados (serão preenchidos em _preparar_dados)
        self.XTreino = None
        self.yTreino = None
        self.XVal = None
        self.yVal = None
        self.XTeste = None
        self.yTeste = None
    
    def _preparar_dados(self, test_size_final=0.3, val_size=0.2):
        """Método privado para carregar e dividir os dados."""
        print("Preparando dados para Adaline Logístico (NumPy)...")
        bancoCompleto = pd.read_csv(self.caminho_arquivo)
        X = bancoCompleto.drop('Doença Cardíaca', axis=1)
        y = bancoCompleto['Doença Cardíaca']

        # Divisão principal usando sua função (ex: 70% treino/val, 30% teste)
        XTreinoVal, self.XTeste, yTreinoVal, self.yTeste = meuTrainTestSplit(
            X, y, testeSize=test_size_final, randomState=self.random_state
        )

        # Divisão secundária para criar conjunto de validação (ex: 80% treino, 20% val de XTreinoVal)
        self.XTreino, self.XVal, self.yTreino, self.yVal = train_test_split(
            XTreinoVal, yTreinoVal, test_size=val_size, random_state=self.random_state, stratify=yTreinoVal
        )
        print("Dados preparados.")
        
    def _treinar(self, n_epocas=30000, taxa_aprendizado=0.0001):
        """Método privado para instanciar e treinar o modelo Adaline NumPy."""
        if self.XTreino is None:
            raise ValueError("Os dados precisam ser preparados antes do treino. Chame _preparar_dados().")

        print(f"\nIniciando treinamento Adaline (NumPy) por {n_epocas} épocas...")
        self.modelo = AdalineLogistica(
            taxaAprendizado=taxa_aprendizado,
            nEpocas=n_epocas,
            estagioAleatorio=self.random_state
        )
        # Executa o treinamento (convertendo para NumPy arrays com .values)
        self.modelo.fit(
            self.XTreino.values,
            self.yTreino.values,
            self.XVal.values, # Passando dados de validação para o fit
            self.yVal.values
        )
        print("Treinamento Adaline (NumPy) concluído.")
    
    def _avaliar_e_visualizar(self):
        """Método privado para avaliar o modelo e plotar gráficos."""
        if self.modelo is None or self.XTeste is None:
            raise ValueError("Modelo não treinado ou dados de teste não preparados.")

        print("\nAvaliando Adaline (NumPy) no conjunto de teste:")
        # Fazer previsões
        probabilidades = self.modelo.predictProba(self.XTeste.values)
        yPred = self.modelo.predict(self.XTeste.values)

        # Calcular acurácia
        acuracia = np.mean(yPred == self.yTeste.values) * 100
        print(f"Acurácia do modelo Adaline Logístico (NumPy): {acuracia:.2f}%")

        # Mostrar DataFrame de resultados
        df_resultados = pd.DataFrame({
            'ID_do_Paciente_no_Teste': self.yTeste.index,
            'Realidade_Observada': self.yTeste.values,
            'Chance_de_Doenca_Cardiaca_%': (probabilidades * 100).round(2)
        })
        print("\nVisualização das previsões (Adaline NumPy):")
        print(df_resultados.head())

        # Gerar Gráficos
        print("\nGerando gráficos de análise (Adaline NumPy)...")
        # Usa os métodos estáticos da classe GraficosAdalineLogistica
        GraficosAdalineLogistica.plotarCurvaErro(self.modelo)
        GraficosAdalineLogistica.plotarMatrizConfusao(self.yTeste.values, yPred, classes=['Não Doente', 'Doente'])
        GraficosAdalineLogistica.plotarCurvasAprendizado(self.modelo)
        
    def rodar(self, n_epocas=30000, taxa_aprendizado=0.0001):
        """
        Executa o pipeline completo: prepara os dados, treina o modelo,
        avalia no conjunto de teste e plota os gráficos.
        Parâmetros:
        - n_epocas: Número de épocas para o treinamento.
        - taxa_aprendizado: Taxa de aprendizado (eta) para o treinamento.
        """
        self._preparar_dados() # Usa os tamanhos padrão de teste/validação
        self._treinar(n_epocas=n_epocas, taxa_aprendizado=taxa_aprendizado)
        self._avaliar_e_visualizar()
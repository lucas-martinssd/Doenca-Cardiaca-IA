#Biblioteca usada para trabalhar com dados em tabelas.
import pandas as pd
import numpy as np
# Cria visualizações e gráficos com os dados
import matplotlib.pyplot as plt
from AdalineLogistica import AdalineLogistica

""" Classe que vai usar as ferramentas feitas na AdalineLogistica para executar a tarefa do início ao fim. """

""" meuTrainTestSplit: Tem a função de dividir o conjunto de dados em dois grupos, um para treinar o modelo e o outro completamente separado, para testá-lo. """
#Essa divisão é feita de forma estratificada, ou seja, deixando a proporção de pacientes com doença cardíaca e sem doença cardíaca sempre a mesma.
#Recebe como parâmetros os features X, o alvo y, o tamanho do grupo de teste testeSize 0.3, ou seja, 30% por ser um valor padrão e a semente aleatória para garantir a reprodutibilidade.
def meuTrainTestSplit(X, y, testeSize=0.3, randomState=42):
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
    """ Loop que vai realizar a divisão dos pacientes de forma igual, ou seja, contendo o mesmo número de pacientes saudáveis e doentes, para treino e para teste. 
        O loop vai ocorrer uma vez para a classe 0 saudáveis e uma para a classe 1 doentes. """
    for cls in classes:
        #Essa linha vai selecionar os índices dos pacientes que pertencem a classe do loop atual, 0 saudável ou 1 doente.
        indicesClasse = indices[y == cls]
        #Embaralha aleatoriamente os índices dentro daquela classe do loop.
        rgen.shuffle(indicesClasse)
        #Calcula a porcentagem de pacientes que vão ser usados para o treino com base no testeSize que já está definido 30%, ou seja o treino é 70%;
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

""" Parte do código que usa os dados já tratados e os utiliza para treinar, testar e avaliar o modelo Adaline Logístico."""
print("Iniciando o treinamento do modelo Adaline Logístico...")
# Carregar os dados do banco completo
caminhoBancoCompleto = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/workspace/open-heart-api/.git/DadosCompletos/BancoCompleto.csv'
bancoCompleto = pd.read_csv(caminhoBancoCompleto)
# Separar Features X e Alvo y
X = bancoCompleto.drop('Doença Cardíaca', axis=1)
y = bancoCompleto['Doença Cardíaca']
# Dividir e escalar os dados usando a função meuTrainTestSplit
XTrain, XTest, yTrain, yTest = meuTrainTestSplit(X, y, testeSize=0.3, randomState=42)
# Treinar o modelo Adaline Logístico usando a classe AdalineLogistica que é o neurônio da nossa rede neural.
adaline = AdalineLogistica(taxaAprendizado=0.01, nEpocas=100)
# Executa o treinamento
adaline.fit(XTrain.values, yTrain.values)
# Fazer previsões e mostrar resultados após o treinamento
probabilidades = adaline.predictProba(XTest.values)
# Faz a classificação final, retornando 0 ou 1, se o paciente tem ou não doença cardíaca.
yPred = adaline.predict(XTest.values)
# Calcula a acurácia do modelo, que é a porcentagem de acertos entre os pacientes do grupo de teste.
acuracia = np.mean(yPred == yTest.values) * 100
# Acurácia do modelo indica a porcentagem entre todos os pacientes no grupo de teste, para quantos deles
# o modelo acertou o diagnóstico
print(f"Acurácia do modelo Adaline Logístico: {acuracia:.2f}%")
# Criar DataFrame de resultados para visualização
df_resultados = pd.DataFrame({
    # Número da linha no banco de dados, que indica qual é o paciente
    'ID_do_Paciente_no_Teste': yTest.index,
    # Indica se o paciente tem ou não doença cardíaca, dado já existente no banco original heart
    'Realidade_Observada': yTest.values,
    # Porcentagem indicada pelo modelo de chance de ter doença cardíaca
    'Chance_de_Doenca_Cardiaca_%': (probabilidades * 100).round(2)
})
print("\nVisualização das previsões:")
print(df_resultados.head())


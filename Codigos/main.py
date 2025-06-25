import pandas as pd
import numpy as np
# Cria visualizações e gráficos com os dados
import matplotlib.pyplot as plt
from AdalineLogistica import AdalineLogistica

def meuTrainTestSplit(X, y, testeSize=0.3, randomState=42):
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    indices = np.arange(len(y))
    classes = np.unique(y)
    indicesTreino, indicesTeste = [], []
    rgen = np.random.RandomState(randomState)
    for cls in classes:
        indicesClasse = indices[y == cls]
        rgen.shuffle(indicesClasse)
        pontoCorte = int(len(indicesClasse) * (1 - testeSize))
        indicesTreino.extend(indicesClasse[:pontoCorte])
        indicesTeste.extend(indicesClasse[pontoCorte:])
    rgen.shuffle(indicesTreino)
    rgen.shuffle(indicesTeste)
    XTrain = X.iloc[indicesTreino]
    XTest = X.iloc[indicesTeste]
    yTrain = y.iloc[indicesTreino]
    yTest = y.iloc[indicesTeste]
    return XTrain, XTest, yTrain, yTest

class MeuStandardScaler:
    def __init__(self):
        self.media = None
        self.desvioPadrao = None
    def fit(self, X):
        self.media = np.mean(X, axis=0)
        self.desvioPadrao = np.std(X, axis=0)
        return self
    def transform(self, X):
        epsilon = 1e-15
        return (X - self.media) / (self.desvioPadrao + epsilon)
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
print("Iniciando o treinamento do modelo Adaline Logístico...")
# Carregar os dados do banco completo
caminhoBancoCompleto = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/workspace/open-heart-api/.git/DadosCompletos/BancoCompleto.csv'
bancoCompleto = pd.read_csv(caminhoBancoCompleto)
# Separar Features X e Alvo y
X = bancoCompleto.drop('Doença Cardíaca', axis=1)
y = bancoCompleto['Doença Cardíaca']
# Dividir e escalar os dados
XTrain, XTest, yTrain, yTest = meuTrainTestSplit(X, y, testeSize=0.3, randomState=42)
scaler = MeuStandardScaler()
XTrainScaled = scaler.fit_transform(XTrain.values)
XTestScaled = scaler.transform(XTest.values)
# Treinar o modelo Adaline Logístico
adaline = AdalineLogistica(taxaAprendizado=0.01, nEpocas=100)
adaline.fit(XTrainScaled, yTrain.values)
# Fazer previsões e mostrar resultados
probabilidades = adaline.predictProba(XTestScaled)
yPred = adaline.predict(XTestScaled)
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


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


class GraficosAdalineLogistica:
    @staticmethod
    def plotarCurvaErro(modelo):
        """
        Plota a evolução do Erro Quadrático Médio do modelo a cada época.
        """
        if not hasattr(modelo, 'erros') or not modelo.erros:
            print("Aviso: O modelo não possui dados de erro para plotar.")
            return
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(modelo.erros) + 1), modelo.erros, marker='o', color='#0000FF', linestyle='-')
        plt.title('Evolução do Erro por Época', fontsize=16, fontweight='bold')
        plt.xlabel('Épocas', fontsize=12)
        plt.ylabel('Erro', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    """Plota a matriz de confusão para avaliar a performance."""
    @staticmethod
    def plotarMatrizConfusao(yReal, previsoes, classes=['Não Doente', 'Doente']):
        matriz = confusion_matrix(yReal, previsoes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"size": 14})
        plt.title('Matriz de Confusão', fontsize=16, fontweight='bold')
        plt.xlabel('Valor Verdadeiro', fontsize=12)
        plt.ylabel('Valor Previsto', fontsize=12)
        plt.show()
        
    """Plota as curvas de acurácia de treino e validação."""
    @staticmethod
    def plotarCurvasAprendizado(modelo):
        if not modelo.historicoAcuraciaVal:
            print("Aviso: Dados de validação não foram fornecidos durante o treino. Plotando apenas a acurácia de treino.")
        plt.figure(figsize=(10, 6))
        epocas = range(1, len(modelo.historicoAcuraciaTreino) + 1)
        plt.plot(epocas, modelo.historicoAcuraciaTreino, marker='.', color='#0000FF', label='Acurácia Treino')
        if modelo.historicoAcuraciaVal:
            plt.plot(epocas, modelo.historicoAcuraciaVal, marker='.', color='orange', label='Acurácia Validação')
        plt.title('Curvas de Aprendizado', fontsize=16, fontweight='bold')
        plt.xlabel('Épocas', fontsize=12)
        plt.ylabel('Acurácia', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(0, 1.05)
        plt.show()
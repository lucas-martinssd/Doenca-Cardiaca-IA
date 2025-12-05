#Biblioteca usada para trabalhar com dados em tabelas.
import pandas as pd
from sklearn.model_selection import train_test_split
from AdalineLogistica.ExecutorAdalineLogistica import ExecutorAdalineLogistica
from Multicamadas.ExecutorMulticamadas import ExecutorMulticamadas
from MulticamadasBiblioteca.ExecutorMulticamadasBibliotecas import ExecutorMulticamadasBibliotecas
from AdalineLogisticaBibliotecas.ExecutorAdalineLogisticaBibliotecas import ExecutorAdalineLogisticaBibliotecas


# Função para Multicamadas NumPy (se ExecutorMulticamadas não carregar/dividir)
def carregar_e_dividir_dados_numpy(caminho_arquivo):
    """
    Carrega o dataset, separa as features (X) do alvo (y) e divide os dados
    para o ExecutorMulticamadas NumPy (ex: 80% treino/val, 20% teste).
    Adapte conforme a necessidade do ExecutorMulticamadas.
    """
    bancoCompleto = pd.read_csv(caminho_arquivo)
    X = bancoCompleto.drop('Doença Cardíaca', axis=1)
    y = bancoCompleto['Doença Cardíaca']
    # Usando a divisão 80/20 que parecia ser usada no bloco original do Multicamadas
    xTreinoVal, xTeste, yTreinoVal, yTeste = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Retorna DataFrames/Series Pandas, pois o Executor NumPy provavelmente espera isso
    return xTreinoVal, yTreinoVal, xTeste, yTeste

# --- Funções de Execução para Cada Modelo ---

def executar_adaline_numpy():
    """Executa o modelo Adaline Logístico implementado com NumPy."""
    print("--- Iniciando Execução: Adaline Logístico (NumPy) ---")
    caminhoBancoCompleto = r'C:\Users\Lucas\Documents\TCC - Previsao Insuficiencia Cardiaca\Doenca-Cardiaca-IA\Dados\DadosCompletos\BancoCompleto.csv'
    try:
        executor_adaline_np = ExecutorAdalineLogistica(
            caminho_arquivo=caminhoBancoCompleto
        )
        executor_adaline_np.rodar(
            n_epocas=30000,
            taxa_aprendizado=0.0001
        )
    except Exception as e:
        print(f"Erro durante a execução do Adaline NumPy: {e}")
    finally:
        print("--- Fim da Execução: Adaline Logístico (NumPy) ---\n")

def executar_multicamadas_numpy():
    """Executa o modelo Multicamadas implementado com NumPy."""
    print("--- Iniciando Execução: Multicamadas (NumPy) ---")
    caminhoBanco = r'C:\Users\Lucas\Documents\TCC - Previsao Insuficiencia Cardiaca\Doenca-Cardiaca-IA\Dados\DadosCompletos\BancoCompleto.csv'
    try:
        # Carregar/dividir dados como ExecutorMulticamadas espera
        xTreinoVal, yTreinoVal, xTeste, yTeste = carregar_e_dividir_dados_numpy(caminhoBanco)

        executorMlp = ExecutorMulticamadas(xTreinoVal, yTreinoVal, xTeste, yTeste)
        executorMlp.rodar() # Certifique-se que os hiperparâmetros desejados estão definidos aqui ou dentro de ExecutorMulticamadas
    except Exception as e:
        print(f"Erro durante a execução do Multicamadas NumPy: {e}")
    finally:
        print("--- Fim da Execução: Multicamadas (NumPy) ---\n")

def executar_multicamadas_keras(n_epocas=30000, taxa_aprendizado=0.0001):
    """Executa o modelo Multicamadas usando Keras/TensorFlow."""
    print("--- Iniciando Execução: Multicamadas (Keras) ---")
    caminhoBanco = r'C:\Users\Lucas\Documents\TCC - Previsao Insuficiencia Cardiaca\Doenca-Cardiaca-IA\Dados\DadosCompletos\BancoCompleto.csv'
    try:
        executor_keras_mlp = ExecutorMulticamadasBibliotecas(
            caminho_arquivo=caminhoBanco,
            n_oculta=10
        )
        executor_keras_mlp.rodar(
            n_epocas=n_epocas,
            taxa_aprendizado=taxa_aprendizado
        )
    except Exception as e:
        print(f"Erro durante a execução do Multicamadas Keras: {e}")
    finally:
        print("--- Fim da Execução: Multicamadas (Keras) ---\n")

def executar_adaline_keras(n_epocas=30000, taxa_aprendizado=0.0001):
    """Executa o modelo Adaline Logístico usando Keras/TensorFlow."""
    print("--- Iniciando Execução: Adaline Logístico (Keras) ---")
    caminhoBanco = r'C:\Users\Lucas\Documents\TCC - Previsao Insuficiencia Cardiaca\Doenca-Cardiaca-IA\Dados\DadosCompletos\BancoCompleto.csv'
    try:
        executor_adaline_keras = ExecutorAdalineLogisticaBibliotecas(
            caminho_arquivo=caminhoBanco
        )
        executor_adaline_keras.rodar(
            n_epocas=n_epocas,
            taxa_aprendizado=taxa_aprendizado
        )
    except Exception as e:
        print(f"Erro durante a execução do Adaline Keras: {e}")
    finally:
        print("--- Fim da Execução: Adaline Keras ---\n")
        
# --- Bloco Principal ÚNICO ---
if __name__ == "__main__":

    # CONFERIR SE O OS GRÁFICOS ESTÁO COMPATIVEIS

    # --- ESCOLHA QUAL(IS) MODELO(S) EXECUTAR ---
    # Descomente a(s) linha(s) do(s) modelo(s) que você quer rodar.

    # 1. Adaline Logístico (Implementação NumPy)
    executar_adaline_numpy()
    
    # 4. Adaline Logístico (Usando Keras/TensorFlow)
    executar_adaline_keras() # Exemplo: rodando Adaline Keras com MSE

    # 2. Multicamadas (Implementação NumPy)
    executar_multicamadas_numpy()

    # 3. Multicamadas (Usando Keras/TensorFlow)
    executar_multicamadas_keras() # Exemplo com BCE e outros hiperparâmetros

    
#Biblioteca usada para trabalhar com dados em tabelas.
import pandas as pd
from sklearn.model_selection import train_test_split
from AdalineLogistica.ExecutorAdalineLogistica import ExecutorAdalineLogistica
from Multicamadas.ExecutorMulticamadas import ExecutorMulticamadas
from MulticamadasBiblioteca.ExecutorMulticamadasBibliotecas import ExecutorMulticamadasBibliotecas
from AdalineLogisticaBibliotecas.ExecutorAdalineLogisticaBibliotecas import ExecutorAdalineLogisticaBibliotecas


#Caminhos Bancos
caminho_base = r'C:\Users\Lucas\Documents\TCC - Previsao Insuficiencia Cardiaca\Doenca-Cardiaca-IA\Dados\DadosCompletos'
BANCO_1 = f'{caminho_base}\BancoCompleto1.csv'
BANCO_2 = f'{caminho_base}\BancoCompleto2.csv'

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

def executar_adaline_numpy(caminho_csv, nome_banco):
    """Executa o modelo Adaline Logístico implementado com NumPy."""
    print(f"--- Iniciando Execução: Adaline Logístico (NumPy) | {nome_banco} ---")
    try:
        executor_adaline_np = ExecutorAdalineLogistica(caminho_arquivo=caminho_csv)
        executor_adaline_np.rodar(
            n_epocas=3000,
            taxa_aprendizado=0.0001
        )
    except Exception as e:
        print(f"Erro durante a execução do Adaline NumPy: ({nome_banco}): {e}")
    finally:
        print(f"--- Fim da Execução: Adaline Logístico (NumPy) | {nome_banco} ---\n")

def executar_multicamadas_numpy(caminho_csv, nome_banco):
    """Executa o modelo Multicamadas implementado com NumPy."""
    print(f"--- Iniciando Execução: Multicamadas (NumPy) | {nome_banco} ---")
    try:
        # Carregar/dividir dados como ExecutorMulticamadas espera
        xTreinoVal, yTreinoVal, xTeste, yTeste = carregar_e_dividir_dados_numpy(caminho_csv)
        executorMlp = ExecutorMulticamadas(xTreinoVal, yTreinoVal, xTeste, yTeste)
        executorMlp.rodar() # Certifique-se que os hiperparâmetros desejados estão definidos aqui ou dentro de ExecutorMulticamadas
    except Exception as e:
        print(f"Erro durante a execução do Multicamadas NumPy: ({nome_banco}): {e}")
    finally:
        print(f"--- Fim da Execução: Multicamadas (NumPy) | {nome_banco} ---\n")

def executar_multicamadas_keras(caminho_csv, nome_banco, n_epocas=3000, taxa_aprendizado=0.0001):
    """Executa o modelo Multicamadas usando Keras/TensorFlow."""
    print(f"--- Iniciando Execução: Multicamadas (Keras) | {nome_banco} ---")
    try:
        executor_keras_mlp = ExecutorMulticamadasBibliotecas(caminho_arquivo=caminho_csv, n_oculta=10)
        executor_keras_mlp.rodar(
            n_epocas=n_epocas,
            taxa_aprendizado=taxa_aprendizado
        )
    except Exception as e:
        print(f"Erro durante a execução do Multicamadas Keras: ({nome_banco}): {e}")
    finally:
        print(f"--- Fim da Execução: Multicamadas (Keras) | {nome_banco} ---\n")

def executar_adaline_keras(caminho_csv, nome_banco, n_epocas=3000, taxa_aprendizado=0.0001):
    """Executa o modelo Adaline Logístico usando Keras/TensorFlow."""
    print(f"--- Iniciando Execução: Adaline Logístico (Keras) | {nome_banco} ---")
    try:
        executor_adaline_keras = ExecutorAdalineLogisticaBibliotecas(caminho_arquivo=caminho_csv)
        executor_adaline_keras.rodar(
            n_epocas=n_epocas,
            taxa_aprendizado=taxa_aprendizado
        )
    except Exception as e:
        print(f"Erro durante a execução do Adaline Keras: ({nome_banco}): {e}")
    finally:
        print(f"--- Fim da Execução: Adaline Keras | {nome_banco} ---\n")
        
# --- Bloco Principal ÚNICO ---
if __name__ == "__main__":

    # CONFERIR SE O OS GRÁFICOS ESTÁO COMPATIVEIS

    # --- ESCOLHA QUAL(IS) MODELO(S) EXECUTAR ---
    # Descomente a(s) linha(s) do(s) modelo(s) que você quer rodar.

    # --- EXECUÇÃO BANCO 1 (UCI Heart) ---
    print("==========================================")
    print("    EXECUTANDO EXPERIMENTOS NO BANCO 1")
    print("==========================================\n")
    
    #Sem Biblioteca
    executar_adaline_keras(BANCO_1, "Banco 1")
    executar_multicamadas_keras(BANCO_1, "Banco 1")
    
    # Com Biblioteca
    executar_adaline_numpy(BANCO_1, "Banco 1")
    executar_multicamadas_numpy(BANCO_1, "Banco 1")

    # --- EXECUÇÃO BANCO 2 (Heart Failure Clinical) ---
    print("==========================================")
    print("    EXECUTANDO EXPERIMENTOS NO BANCO 2")
    print("==========================================\n")
    
    #Sem Biblioteca
    executar_adaline_keras(BANCO_2, "Banco 2")
    executar_multicamadas_keras(BANCO_2, "Banco 2")
    
    # Com Biblioteca
    executar_adaline_numpy(BANCO_2, "Banco 2")
    executar_multicamadas_numpy(BANCO_2, "Banco 2")
    
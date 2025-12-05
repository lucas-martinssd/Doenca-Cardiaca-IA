import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from MulticamadasBiblioteca.MulticamadasBibliotecas import MulticamadasBibliotecas
from Graficos.GraficosAdalineLogistica import GraficosAdalineLogistica

class ExecutorMulticamadasBibliotecas:
    """
    Orquestra o carregamento, pré-processamento, treinamento, avaliação
    e visualização do modelo MulticamadasBibliotecas (Keras).
    Equivalente à sua classe 'ExecutorMulticamadas' anterior.
    """
    def __init__(self, caminho_arquivo, n_oculta=10, random_state=42):
        self.caminho_arquivo = caminho_arquivo
        self.n_oculta = n_oculta
        self.random_state = random_state
        self.scaler = StandardScaler() # Guardar o scaler para usar nos dados de teste/validação
        self.modelo = None
        self.historico_treino = None # Para guardar as métricas de cada época
        # Atributos para os dados (serão preenchidos em _preparar_dados)
        self.XTreino_tf = None
        self.yTreino_tf = None
        self.XVal_tf = None
        self.yVal_tf = None
        self.XTeste_tf = None
        self.yTeste_tf = None

    def _preparar_dados(self, test_size_final=0.2, val_size=0.2):
        """Método privado para carregar, dividir e escalar os dados."""
        print("Preparando dados...")
        bancoCompleto = pd.read_csv(self.caminho_arquivo)
        X = bancoCompleto.drop('Doença Cardíaca', axis=1)
        y = bancoCompleto['Doença Cardíaca']

        XTreinoVal, XTeste, yTreinoVal, yTeste = train_test_split(
            X, y, test_size=test_size_final, random_state=self.random_state, stratify=y
        )
        XTreino, XVal, yTreino, yVal = train_test_split(
            XTreinoVal, yTreinoVal, test_size=val_size, random_state=self.random_state, stratify=yTreinoVal
        )

        # Escalonar os dados
        XTreino_scaled = self.scaler.fit_transform(XTreino) # Ajusta e transforma no treino
        XVal_scaled = self.scaler.transform(XVal)       # Só transforma na validação
        XTeste_scaled = self.scaler.transform(XTeste)     # Só transforma no teste

        # Converter para tensores do TensorFlow
        self.XTreino_tf = tf.convert_to_tensor(XTreino_scaled, dtype=tf.float32)
        self.yTreino_tf = tf.convert_to_tensor(yTreino.values.reshape(-1, 1), dtype=tf.float32)
        self.XVal_tf = tf.convert_to_tensor(XVal_scaled, dtype=tf.float32)
        self.yVal_tf = tf.convert_to_tensor(yVal.values.reshape(-1, 1), dtype=tf.float32)
        self.XTeste_tf = tf.convert_to_tensor(XTeste_scaled, dtype=tf.float32)
        self.yTeste_tf = tf.convert_to_tensor(yTeste.values.reshape(-1, 1), dtype=tf.float32)
        print("Dados preparados e escalonados.")

    def _treinar(self, n_epocas=100, taxa_aprendizado=0.001):
        """Método privado para instanciar e treinar o modelo Keras."""
        if self.XTreino_tf is None:
            raise ValueError("Os dados precisam ser preparados antes do treino. Chame _preparar_dados().")

        n_entrada = self.XTreino_tf.shape[1]
        n_saida = 1 # Classificação Binária

        # Instanciar o modelo
        self.modelo = MulticamadasBibliotecas(n_entrada, self.n_oculta, n_saida, self.random_state)

        # Compilar o modelo: define a função de custo, otimizador e métricas
        self.modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=taxa_aprendizado),
            loss=keras.losses.MeanSquaredError(), # <--- Mudado para MSE
            metrics=[
                keras.metrics.BinaryAccuracy(name='accuracy'), # Mantém a acurácia
                keras.metrics.MeanSquaredError(name='mse')     # Adiciona MSE como métrica explícita (opcional)
            ]
        )
        
        callback_parada = keras.callbacks.EarlyStopping(
            monitor='val_loss', # Métrica a ser monitorada
            patience=20,        # Nº de épocas sem melhora antes de parar
            restore_best_weights=True # Volta aos melhores pesos encontrados
        )

        print(f"\nIniciando treinamento por {n_epocas} épocas...")
        # Treinar o modelo usando o método fit() do Keras
        # validation_data permite acompanhar a performance na validação a cada época
        self.historico_treino = self.modelo.fit(
            self.XTreino_tf,
            self.yTreino_tf,
            epochs=n_epocas,
            validation_data=(self.XVal_tf, self.yVal_tf),
            verbose=1 # Mostra 0 linha por época
        )
        print("Treinamento concluído.")

    def _avaliar_e_visualizar(self):
        """Avalia o modelo e plota os gráficos padronizados usando GraficosAdalineLogistica."""
        if self.modelo is None or self.XTeste_tf is None:
            raise ValueError("Modelo não treinado ou dados de teste não preparados.")

        print("\nAvaliando Multicamadas (Keras) no conjunto de teste:")
        
        results = self.modelo.evaluate(self.XTeste_tf, self.yTeste_tf, verbose=0)
        loss_teste = results[0] # MSE
        acc_teste = results[1]  # Acurácia
        
        print(f"Erro Quadrático Médio (MSE) no teste: {loss_teste:.4f}")
        print(f"Acurácia no teste: {acc_teste*100:.2f}%")

        # --- A "PONTE" (Mock Object) ---
        # 1. Pegar dados do Keras
        history_dict = self.historico_treino.history
        
        # 2. Criar um objeto "adaptador"
        class MockModelGrafico:
            def __init__(self, history):
                # Traduz 'loss' (MSE) do Keras para 'erros' do NumPy
                self.erros = history.get('loss')
                
                # Traduz 'accuracy' do Keras para 'historicoAcuraciaTreino'
                self.historicoAcuraciaTreino = history.get('accuracy') or history.get('binary_accuracy')
                
                # Traduz 'val_accuracy' do Keras para 'historicoAcuraciaVal'
                self.historicoAcuraciaVal = history.get('val_accuracy') or history.get('val_binary_accuracy')

        # 3. Criar o objeto com os dados
        mock_modelo = MockModelGrafico(history_dict)
        
        # --- PREVISÕES PARA A MATRIZ ---
        y_proba = self.modelo.predict(self.XTeste_tf)
        y_pred = (y_proba >= 0.5).astype(int).flatten()
        y_teste_numpy = self.yTeste_tf.numpy().flatten()
        classes = ['Não Doente', 'Doente']

        # --- 4. CHAMAR A CLASSE DE GRÁFICOS PADRONIZADA ---
        print("\nGerando gráficos de análise (Multicamadas Keras)...")
        
        # Gráfico 1: Curva de Erro
        GraficosAdalineLogistica.plotarCurvaErro(mock_modelo)
        
        # Gráfico 2: Matriz de Confusão
        # (Note que passamos os eixos corretos que sua classe espera)
        GraficosAdalineLogistica.plotarMatrizConfusao(y_teste_numpy, y_pred, classes=classes)

        # Gráfico 3: Curvas de Aprendizado (Acurácia)
        GraficosAdalineLogistica.plotarCurvasAprendizado(mock_modelo)

    def rodar(self, n_epocas=1000, taxa_aprendizado=0.0001):
        """Executa o pipeline completo: prepara, treina, avalia e visualiza."""
        self._preparar_dados()
        self._treinar(n_epocas=n_epocas, taxa_aprendizado=taxa_aprendizado)
        self._avaliar_e_visualizar()
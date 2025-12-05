import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from AdalineLogisticaBibliotecas.AdalineLogisticaBibliotecas import AdalineLogisticaBibliotecas
from Graficos.GraficosAdalineLogistica import GraficosAdalineLogistica

class ExecutorAdalineLogisticaBibliotecas:
    """
    Orquestra o carregamento, pré-processamento, treinamento, avaliação
    e visualização do modelo AdalineLogisticaBibliotecas (Keras).
    """
    def __init__(self, caminho_arquivo, random_state=42):
        self.caminho_arquivo = caminho_arquivo
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.modelo = None
        self.historico_treino = None
        self.XTreino_tf = None
        self.yTreino_tf = None
        self.XVal_tf = None
        self.yVal_tf = None
        self.XTeste_tf = None
        self.yTeste_tf = None

    def _preparar_dados(self, test_size_final=0.2, val_size=0.2):
        """Método privado para carregar, dividir e escalar os dados."""
        # (Este método pode ser exatamente igual ao da classe ExecutorMulticamadasBibliotecas)
        print("Preparando dados para Adaline Logístico (Keras)...")
        bancoCompleto = pd.read_csv(self.caminho_arquivo)
        X = bancoCompleto.drop('Doença Cardíaca', axis=1)
        y = bancoCompleto['Doença Cardíaca']
        XTreinoVal, XTeste, yTreinoVal, yTeste = train_test_split(
            X, y, test_size=test_size_final, random_state=self.random_state, stratify=y
        )
        XTreino, XVal, yTreino, yVal = train_test_split(
            XTreinoVal, yTreinoVal, test_size=val_size, random_state=self.random_state, stratify=yTreinoVal
        )
        XTreino_scaled = self.scaler.fit_transform(XTreino)
        XVal_scaled = self.scaler.transform(XVal)
        XTeste_scaled = self.scaler.transform(XTeste)
        self.XTreino_tf = tf.convert_to_tensor(XTreino_scaled, dtype=tf.float32)
        self.yTreino_tf = tf.convert_to_tensor(yTreino.values.reshape(-1, 1), dtype=tf.float32)
        self.XVal_tf = tf.convert_to_tensor(XVal_scaled, dtype=tf.float32)
        self.yVal_tf = tf.convert_to_tensor(yVal.values.reshape(-1, 1), dtype=tf.float32)
        self.XTeste_tf = tf.convert_to_tensor(XTeste_scaled, dtype=tf.float32)
        self.yTeste_tf = tf.convert_to_tensor(yTeste.values.reshape(-1, 1), dtype=tf.float32)
        print("Dados preparados e escalonados.")
        
    def _treinar(self, n_epocas=100, taxa_aprendizado=0.001):
        """Método privado para instanciar e treinar o modelo Adaline Keras."""
        if self.XTreino_tf is None:
            raise ValueError("Os dados precisam ser preparados antes do treino. Chame _preparar_dados().")

        n_entrada = self.XTreino_tf.shape[1]
        n_saida = 1 # Saída binária

        self.modelo = AdalineLogisticaBibliotecas(n_entrada, n_saida, self.random_state)

        # Função de custo fixada como MSE (Erro)
        loss_func = keras.losses.MeanSquaredError()
        self._loss_name_plot = 'Erro Quadrático Médio (MSE)' # Salva o nome para os prints
        print("INFO: Usando MeanSquaredError (MSE) como função de custo (Erro).")

        # Compilar o modelo
        self.modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=taxa_aprendizado),
            loss=loss_func,
            metrics=[keras.metrics.BinaryAccuracy(name='accuracy')] # Acurácia como métrica
        )

        print(f"\nIniciando treinamento Adaline por {n_epocas} épocas...")
        self.historico_treino = self.modelo.fit(
            self.XTreino_tf,
            self.yTreino_tf,
            epochs=n_epocas,
            validation_data=(self.XVal_tf, self.yVal_tf),
            verbose=1 # Mostrar barra de progresso
            # callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)] # Early stopping opcional
        )
        print("Treinamento Adaline concluído.")
    
    def _avaliar_e_visualizar(self):
        """Avalia o modelo e plota os gráficos padronizados usando GraficosAdalineLogistica."""
        if self.modelo is None or self.XTeste_tf is None:
            raise ValueError("Modelo não treinado ou dados de teste não preparados.")

        print("\nAvaliando Adaline (Keras) no conjunto de teste:")
        
        results = self.modelo.evaluate(self.XTeste_tf, self.yTeste_tf, verbose=0)
        loss_teste = results[0] # MSE
        acc_teste = results[1]  # Acurácia
        
        loss_label = getattr(self, '_loss_name_plot', 'Perda/Erro')
        print(f"{loss_label} no teste: {loss_teste:.4f}")
        print(f"Acurácia no teste: {acc_teste*100:.2f}%")

        # --- A "PONTE" (Mock Object) ---
        history_dict = self.historico_treino.history
        
        class MockModelGrafico:
            def __init__(self, history):
                self.erros = history.get('loss')
                self.historicoAcuraciaTreino = history.get('accuracy') or history.get('binary_accuracy')
                self.historicoAcuraciaVal = history.get('val_accuracy') or history.get('val_binary_accuracy')

        mock_modelo = MockModelGrafico(history_dict)
        
        # --- PREVISÕES PARA A MATRIZ ---
        y_proba = self.modelo.predict(self.XTeste_tf)
        y_pred = (y_proba >= 0.5).astype(int).flatten()
        y_teste_numpy = self.yTeste_tf.numpy().flatten()
        classes = ['Não Doente', 'Doente']

        # --- 4. CHAMAR A CLASSE DE GRÁFICOS PADRONIZADA ---
        print("\nGerando gráficos de análise (Adaline Keras)...")
        
        GraficosAdalineLogistica.plotarCurvaErro(mock_modelo)
        GraficosAdalineLogistica.plotarMatrizConfusao(y_teste_numpy, y_pred, classes=classes)
        GraficosAdalineLogistica.plotarCurvasAprendizado(mock_modelo)

    def rodar(self, n_epocas=100, taxa_aprendizado=0.001):
        """Executa o pipeline completo: prepara, treina, avalia e visualiza."""
        self._preparar_dados()
        self._treinar(n_epocas=n_epocas, taxa_aprendizado=taxa_aprendizado)
        self._avaliar_e_visualizar()
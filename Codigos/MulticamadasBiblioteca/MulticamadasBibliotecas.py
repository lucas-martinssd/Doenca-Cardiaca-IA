import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class MulticamadasBibliotecas(keras.Model):
    """
    Representa a arquitetura da Rede Neural Multicamadas usando Keras.
    Equivalente à sua classe 'Multicamadas' anterior, mas usando camadas Keras.
    """
    def __init__(self, n_entrada, n_oculta, n_saida, random_state=42):
        """
        Construtor da classe.
        Parâmetros:
        - n_entrada: Número de neurônios na camada de entrada (features).
        - n_oculta: Número de neurônios na camada oculta.
        - n_saida: Número de neurônios na camada de saída (1 para binário).
        - random_state: Semente para reprodutibilidade.
        """
        super().__init__()
        tf.random.set_seed(random_state) # Define a semente para Keras/TensorFlow

        # Camada Oculta com ativação ReLU
        self.camada_oculta = keras.layers.Dense(
            units=n_oculta,
            activation='relu',
            input_shape=(n_entrada,), # Define a forma da entrada apenas na primeira camada
            name='camada_oculta' # Nome opcional
        )

        # Camada de Saída com ativação Sigmoid (para classificação binária)
        self.camada_saida = keras.layers.Dense(
            units=n_saida,
            activation='sigmoid',
            name='camada_saida' # Nome opcional
        )

    def call(self, inputs, training=False):
        """
        Define como os dados fluem pela rede (forward pass).
        'training' é um argumento padrão em Keras, útil para camadas como Dropout/BatchNormalization.
        """
        x = self.camada_oculta(inputs)
        return self.camada_saida(x)
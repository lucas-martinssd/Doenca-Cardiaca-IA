import tensorflow as tf
from tensorflow import keras

class AdalineLogisticaBibliotecas(keras.Model):
    """
    Representa a arquitetura do Adaline Logístico (equivalente a Regressão Logística)
    usando Keras. É uma rede neural com apenas uma camada de saída.
    """
    def __init__(self, n_entrada, n_saida=1, random_state=42):
        """
        Construtor da classe.
        Parâmetros:
        - n_entrada: Número de neurônios na camada de entrada (features).
        - n_saida: Número de neurônios na camada de saída (geralmente 1 para binário).
        - random_state: Semente para reprodutibilidade.
        """
        super().__init__()
        tf.random.set_seed(random_state) # Define a semente para Keras/TensorFlow

        # Camada de Saída: Uma única camada Dense com ativação Sigmoid
        # Ela conecta diretamente a entrada (n_entrada) à saída (n_saida)
        self.camada_saida = keras.layers.Dense(
            units=n_saida,
            activation='sigmoid',
            input_shape=(n_entrada,), # Define a forma da entrada
            name='camada_saida_logistica'
        )
        
    def call(self, inputs, training=False):
        """
        Define como os dados fluem pela rede (forward pass).
        Neste caso, a entrada vai direto para a camada de saída.
        """
        return self.camada_saida(inputs)
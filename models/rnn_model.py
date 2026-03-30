"""
models/rnn_model.py
===================
Modèle LSTM pour la prédiction de séries temporelles (T+1).
Architecture : LSTM empilé → Dense → sortie scalaire
"""

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable()
class WeatherLSTM(tf.keras.Model):
    """
    Modèle LSTM pour la prédiction météorologique (T+1).

    Architecture :
      LSTM(128, return_sequences=True)
      → Dropout(0.2)
      → LSTM(64, return_sequences=False)   ← dernier état caché uniquement
      → Dropout(0.2)
      → Dense(32, relu)
      → Dense(1)                           ← prédiction scalaire T+1
    """

    def __init__(self, lstm_units_1: int = 128, lstm_units_2: int = 64,
                 dense_units: int = 32, dropout_rate: float = 0.2):
        super(WeatherLSTM, self).__init__()

        self.lstm_units_1  = lstm_units_1
        self.lstm_units_2  = lstm_units_2
        self.dense_units   = dense_units
        self.dropout_rate  = dropout_rate

        # ── Couche 1 : LSTM avec séquences complètes ──────────────────
        self.lstm1   = layers.LSTM(lstm_units_1, return_sequences=True,
                                   name="lstm_1")
        self.drop1   = layers.Dropout(dropout_rate, name="dropout_1")

        # ── Couche 2 : LSTM — return_sequences=False (dernier état) ───
        self.lstm2   = layers.LSTM(lstm_units_2, return_sequences=False,
                                   name="lstm_2")
        self.drop2   = layers.Dropout(dropout_rate, name="dropout_2")

        # ── Classifieur final ─────────────────────────────────────────
        self.dense1  = layers.Dense(dense_units, activation="relu",
                                    name="dense_1")
        self.out     = layers.Dense(1, name="output")

    def build(self, input_shape):
        # Force la construction de toutes les sous-couches
        self.lstm1.build(input_shape)
        lstm1_out = (input_shape[0], input_shape[1], self.lstm_units_1)
        self.drop1.build(lstm1_out)
        self.lstm2.build(lstm1_out)
        lstm2_out = (input_shape[0], self.lstm_units_2)
        self.drop2.build(lstm2_out)
        self.dense1.build(lstm2_out)
        dense1_out = (input_shape[0], self.dense_units)
        self.out.build(dense1_out)
        super().build(input_shape)

    def call(self, inputs, training: bool = False):
        x = self.lstm1(inputs)
        x = self.drop1(x, training=training)
        x = self.lstm2(x)
        x = self.drop2(x, training=training)
        x = self.dense1(x)
        return self.out(x)

    def get_config(self):
        return {
            "lstm_units_1": self.lstm_units_1,
            "lstm_units_2": self.lstm_units_2,
            "dense_units":  self.dense_units,
            "dropout_rate": self.dropout_rate,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
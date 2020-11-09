from functools import partial

import tensorflow as tf

from tensorflow import keras


ADAM_OPTIMIZER = keras.optimizers.Adam()

EARLY_STOPPING_CB = keras.callbacks.EarlyStopping(
    patience=2,
    restore_best_weights=True
)

class TargetScoringModel(keras.Model):

    RegularizedDense = partial(
        keras.layers.Dense,
        activation='elu',
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(),
        dtype='float64'
    )

    def __init__(self, input_dim, **kwargs):
        super().__init__(**kwargs)
        self.dense_1 = self.RegularizedDense(30, input_shape=input_dim)
        self.dense_2 = self.RegularizedDense(30)
        self.dense_3 = self.RegularizedDense(30)
        self.dense_4 = self.RegularizedDense(30)
        self.score_output = keras.layers.Dense(1, activation='linear', dtype='float64')

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        return self.score_output(x)

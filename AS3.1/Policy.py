import tensorflow as tf
from tensorflow.keras import layers
from keras.optimizers import Adam
from keras.losses import MeanSquaredError


class Policy:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.model = None

    def select_action(self, state):
        pass

    def decay(self):
        """
        Decaying epsilon
        """
        if self.epsilon > 0.01:
            self.epsilon *= 0.99

    def model(self, actions, lr):
        model = tf.keras.Sequential()
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(128, activation="relu"))
        model.compile(optimizer=Adam(learning_rate=lr), loss=MeanSquaredError())

        self.model = model


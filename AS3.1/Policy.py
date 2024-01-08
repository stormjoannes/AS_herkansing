import random
import tensorflow as tf
from tensorflow.keras import layers
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
import numpy as np


class Policy:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.model = None

    def select_action(self, state):
        """
        Implementing a partial random agent.
        This way it sometimes uses the model, but also discovers new paths by the randomness
        How longer the model runs (more trained) the less the chance is to choose a random action because of
        the decaying epsilon.
        """
        random_epsilon = round(random.random(), 2)
        if random_epsilon < self.epsilon:
            # print('select aciton 1')
            action = random.choice((0, 1, 2, 3))
            return action

        else:
            # print('select aciton 2')
            state = np.array([state])
            # print(state, 'state', state.shape)
            output = self.model.predict(state)
            action = np.argmax(output)
            return action

    def decay(self):
        """
        Decaying epsilon
        """
        if self.epsilon > 0.01:
            self.epsilon *= 0.99

    def setup_model(self, dimensions, actions, lr):
        model = tf.keras.Sequential()
        model.add(layers.Dense(64, input_shape=(None, 8)))
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(actions))
        model.compile(optimizer=Adam(learning_rate=lr), loss=MeanSquaredError())

        self.model = model


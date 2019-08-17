from collections import deque
import random
import numpy as np
import keras
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model


class DeepQNetworkAgent:
    def __init__(self):
        self.gamma = 0.95  # Discount rate
        self.epsilon = 0.1  # Exploration rate
        self.learning_rate = 0.0002
        self.memory = deque(maxlen=200)
        self.model = self._build_model()
        self.action_size = 2

    def _build_model(self):
        # Neural Net for Deep-Q learning Model

        # Position P
        input_1 = Input(shape=(8, 8, 1))
        # First layer P
        x1 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(input_1)
        # Second layer P
        x1 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(x1)
        # Part of the third layer
        x1 = Flatten()(x1)

        # Speed V
        input_2 = Input(shape=(8, 8, 1))
        # First layer V
        x2 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(input_2)
        # Second layer V
        x2 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(x2)
        # Part of the third layer
        x2 = Flatten()(x2)

        # Latest traffic signal state L
        input_3 = Input(shape=(2, 1))
        # Part of the third layer
        x3 = Flatten()(input_3)

        x = keras.layers.concatenate([x1, x2, x3])
        # Third layer
        x = Dense(128, activation='relu')(x)
        # Forth layer
        x = Dense(64, activation='relu')(x)
        # Output layer
        x = Dense(2, activation='linear')(x)

        model = Model(inputs=[input_1, input_2, input_3], outputs=[x])
        model.compile(optimizer=keras.optimizers.RMSprop(
            lr=self.learning_rate), loss='mse')

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)

        return np.argmax(act_values[0])  # Returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

from collections import deque
import random
import numpy as np
import keras
import logging
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model


class DeepQNetworkAgent:
    def __init__(self, memory_palace):
        self.gamma = 0.95  # Discount rate
        self.epsilon = 0.1  # Exploration rate
        self.learning_rate = 0.0002
        self.model = self._build_model()
        self.action_size = 2
        self.memory_palace = memory_palace

        logging.info('Gamma: %f' % self.gamma)
        logging.info('Epsilon: %f' % self.epsilon)
        logging.info('Learning Rate: %f' % self.learning_rate)
        logging.info('Memory Palace: %s' % self.memory_palace)

        if self.memory_palace:
            # state/action
            # [[green_north_south/green_east_west, green_north_south/green_north_south],
            # [[green_east_west/green_east_west,     green_east_west/green_north_south]]
            self.memory = [[deque(maxlen=50), deque(maxlen=50)],
                           [deque(maxlen=50), deque(maxlen=50)]]
        else:
            self.memory = deque(maxlen=200)

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
        if self.memory_palace:
            light_state = state[2][0][0][0]
            self.memory[light_state][action].append((state, action, reward, next_state, done))
        else:
            self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)

        return np.argmax(act_values[0])  # Returns action

    def replay(self, batch_size):
        if self.memory_palace:
            # Iterates through every slot of the memory palace
            for state_index in range(2):
                for action_index in range(2):
                    minibatch_size = min(len(self.memory[state_index][action_index]), int(batch_size / 4))
                    minibatch = random.sample(self.memory[state_index][action_index], minibatch_size)

                    for state, action, reward, next_state, done in minibatch:
                        target = reward
                        if not done:
                            prediction = self.model.predict(next_state)
                            target = (reward + self.gamma * np.amax(prediction[0]))
                        target_f = self.model.predict(state)
                        target_f[0][action] = target
                        self.model.fit(state, target_f, epochs=1, verbose=0)
        else:
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target = (reward + self.gamma *
                              np.amax(self.model.predict(next_state)[0]))
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)

    def get_memory_size(self):
        size = 0
        for state_index in range(2):
            for action_index in range(2):
                size += len(self.memory[state_index][action_index])
        return size

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

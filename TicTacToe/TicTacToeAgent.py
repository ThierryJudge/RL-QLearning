from QAgent import QAgent
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
import numpy as np
import random
from TicTacToe.Environment import Environment

from collections import deque
import datetime


class TicTacToeAgent(QAgent):
    def __init__(self, state_size, action_size):
        QAgent.__init__(self, state_size, action_size)

        self.version = 0
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.batch_size = 128
        self.memory_length = 10000

        self.memory = deque(maxlen=self.memory_length)
        print(self.name)

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        # model = Sequential()
        # model.add(Convolution2D(32, kernel_size=(3, 3), padding='same',
        #                         input_shape=(3, 3, 1)))
        # model.add(Activation('relu'))
        # model.add(Convolution2D(32, 3, 3))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.25))
        #
        # model.add(Flatten())
        # model.add(Dense(128))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(self.action_size))
        # model.add(Activation('linear'))
        # model.compile(loss="mse", optimizer=RMSprop())

        return model

    def act(self, state, is_training = True):

        action = np.argmax(self.model.predict(state))

        available_positions = []
        for index, item in enumerate(state):
            if item == 0:
                available_positions.append(index)

        random_action = random.choice(available_positions)

        if np.random.rand(1) < self.epsilon and is_training:
            action = random_action

        return action

    def get_name(self):
        now = datetime.datetime.now()
        return "TicTacToe_{}".format(now.strftime("%Y-%m-%d-%H-%M"))

    def act(self, state, is_training=True):
        state = self.prepocess_state(state)
        return super().act(state, is_training)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def print_memory(self):
        print("Memory")
        for i in self.memory:
            print(i)

    def prepocess_state(self, state):
        #return np.reshape(state, (1, 3, 3, 1))
        return state.reshape(1, 9)

    def update(self):
        batch = random.sample(self.memory, self.batch_size if len(self.memory) > self.batch_size else len(self.memory))
        states = []
        target_qs = []

        for state, action, reward, next_state, done in batch:

            if done:
                target_reward = reward
            else:
                new_q_values = self.model.predict(self.prepocess_state(next_state))
                max_q = np.max(new_q_values)
                target_reward = reward + self.gamma * max_q

            target_q = self.model.predict(self.prepocess_state(state))

            for index, item in enumerate(state.reshape(9)):
                if item != 0:
                    target_q[0, index] = Environment.REWARD_ILLEGAL

            target_q[0, action] = target_reward


            #states.append(self.prepocess_state(state))
            states.append(state.squeeze())
            target_qs.append(target_q.squeeze())

        history = self.model.fit(np.array(states), np.array(target_qs), batch_size=self.batch_size, verbose=0)
        loss = history.history['loss']

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

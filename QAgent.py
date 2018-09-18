from AbstractAgent import AbstractAgent
import tensorflow as tf
import random
import numpy as np
from collections import deque
from matplotlib import pyplot as plt
from tensorflow.core.framework import summary_pb2
import shutil
import os
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class QAgent(AbstractAgent):

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 1.0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.learning_rate_decay = 0.01
        self.batch_size = 128
        self.memory_length = 100000

        self.memory = deque(maxlen=self.memory_length)

        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(self.action_size, input_dim=self.state_size, activation='relu'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def act(self, state, is_training = True):
        action = np.argmax(self.model.predict(state))
        random_action = random.randrange(self.action_size)

        if np.random.rand(1) < self.epsilon and is_training:
            action = random_action

        return action

    def remember(self,state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        batch = random.sample(self.memory, self.batch_size if len(self.memory) > self.batch_size else len(self.memory))
        states = []
        target_qs = []

        for state, action, reward, next_state, done in batch:
            if done:
                target_reward = reward
            else:
                new_q_values = self.model.predict(next_state)
                max_q = np.max(new_q_values)
                target_reward = reward + self.gamma * max_q

            target_q = self.model.predict(state)

            target_q[0, action] = target_reward

            states.append(state.squeeze())
            target_qs.append(target_q.squeeze())

        history = self.model.fit(np.array(states), np.array(target_qs), batch_size=self.batch_size, verbose=0)
        loss = history.history['loss']

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

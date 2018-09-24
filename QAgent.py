from AbstractAgent import AbstractAgent
import tensorflow as tf
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.core.framework import summary_pb2
import datetime
from keras.models import load_model
import os


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

        self.model = self.build_model()

        self.name = self.get_name()

        self.logs_path = 'logs/{}/'.format(self.name)
        self.summary_writer = tf.summary.FileWriter(self.logs_path, graph=tf.get_default_graph())

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.action_size, input_dim=self.state_size, activation='relu'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def act(self, state, is_training = True):
        # if not is_training:
        #     print(self.model.predict(state))
        action = np.argmax(self.model.predict(state))
        random_action = random.randrange(self.action_size)

        if np.random.rand(1) < self.epsilon and is_training:
            action = random_action

        return action

    def remember(self, state, action, reward, next_state, done):
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

    def get_name(self):
        now = datetime.datetime.now()
        return "QAgent_{}".format(now.strftime("%Y-%m-%d-%H-%M"))

    def write_loss_to_tensorboard(self, loss, episode):
        self.write_value_to_tensorboard(loss, 'Loss', episode)
        self.write_value_to_tensorboard(self.epsilon, 'Epsilon', episode)

    def write_score_to_tensorboard(self, score, episode):
        self.write_value_to_tensorboard(score, 'Score', episode)

    def write_value_to_tensorboard(self, value, tag:str, episode:int):
        value = summary_pb2.Summary.Value(tag=tag, simple_value=value)
        summary = summary_pb2.Summary(value=[value])
        self.summary_writer.add_summary(summary, episode)

    def save_model(self):
        s = input("Do you want to save your model? (y/n)")

        while s != 'y' and s != 'n':
            s = input("Do you want to save your model? (y/n)")

        if s == 'y':

            if not os.path.exists('modelsy'):
                os.mkdir('models')

            self.model.save('models/{}.h5'.format(self.get_name()))

    def load_model(self, filepath):
        self.model = load_model(filepath)

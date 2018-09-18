from QAgent import QAgent
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from collections import deque


class CartpoleAgent(QAgent):

    def __init__(self, state_size, action_size):
        QAgent.__init__(self, state_size, action_size)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.batch_size = 32
        self.memory_length = 100000

        self.memory = deque(maxlen=self.memory_length)

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        self.model.add(Dense(48, activation='relu'))
        self.model.add(Dense(self.action_size, activation='linear'))

        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

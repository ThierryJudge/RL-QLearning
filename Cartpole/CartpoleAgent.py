from QAgent import QAgent
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from collections import deque
import datetime


class CartpoleAgent(QAgent):

    def __init__(self, state_size, action_size):
        QAgent.__init__(self, state_size, action_size)

        self.version = 0
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.01
        self.batch_size = 64
        self.memory_length = 100000

        self.memory = deque(maxlen=self.memory_length)
        print(self.name)

    def build_model(self):
        model = Sequential()
        model.add(Dense(48, input_dim=self.state_size, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def get_name(self):
        now = datetime.datetime.now()
        return "Cartpole_{}".format(now.strftime("%Y-%m-%d-%H:%M"))

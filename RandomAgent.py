from AbstractAgent import AbstractAgent
import random


class RandomAgent(AbstractAgent):
    def __init__(self, action_size):
        self.action_size = action_size

    def act(self, state, is_trainig=False):
        return random.randrange(self.action_size)

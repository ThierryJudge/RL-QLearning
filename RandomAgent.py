from AbstractAgent import AbstractAgent
import random


class RandomAgent(AbstractAgent):
    def __init__(self, action_size):
        self.action_size = action_size

    def act(self):
        return random.randrange(self.action_size)

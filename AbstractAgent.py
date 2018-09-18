

class AbstractAgent:

    def act(self, state):
        pass

    def remember(self,state, action, reward, next_state, done):
        pass

    def update(self):
        pass
from RandomAgent import RandomAgent
import random


class SemiRandomAgent(RandomAgent):
    def __init__(self, action_size):
        super().__init__(action_size)

    def act(self, state, is_trainig=False):

        choice = self.check2win(state, 0, 1, 2)
        if choice != -1:
            return choice

        choice = self.check2win(state, 3, 4, 5)
        if choice != -1:
            return choice

        choice = self.check2win(state, 6, 7, 8)
        if choice != -1:
            return choice

        choice = self.check2win(state, 0, 3, 6)
        if choice != -1:
            return choice

        choice = self.check2win(state, 1, 4, 7)
        if choice != -1:
            return choice

        choice = self.check2win(state, 2, 5, 8)
        if choice != -1:
            return choice

        choice = self.check2win(state, 0, 4, 8)
        if choice != -1:
            return choice

        choice = self.check2win(state, 2, 4, 6)
        if choice != -1:
            return choice

        if choice == -1:
            return random.randrange(self.action_size)

    def check2win(self, state, a, b, c):
        #print("Checkwin {}, {}, {}".format(a, b, c))
        if state[a] == 1 or state[b] == 1 or state[c] == 1:
            if state[a] == 1 and state[b] == 1:
                if state[c] == 0:
                    return c
            if state[a] == 1 and state[c] == 1:
                if state[b] == 0:
                    return b
            if state[b] == 1 and state[c] == 1:
                if state[a] == 0:
                    return a
            else:
                return -1

        if state[a] == -1 or state[b] == -1 or state[c] == -11:
            if state[a] == -1 and state[b] == -1:
                if state[c] == 0:
                    return c
            if state[a] == -1 and state[c] == -1:
                if state[b] == 0:
                    return b
            if state[b] == -1 and state[c] == -1:
                if state[a] == 0:
                    return a
            else:
                return -1
        else:
            return -1


if __name__ == '__main__':
    agent = SemiRandomAgent(9)

    board = [-1, 0, 1,
             0, 0, 0,
             1, 1, -1]

    print(agent.act(board))

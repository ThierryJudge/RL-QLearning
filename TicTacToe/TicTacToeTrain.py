from TicTacToe.Environment import Environment
from TicTacToe.TicTacToeAgent import TicTacToeAgent
from QAgent import QAgent
import numpy as np
from TicTacToe.Environment import draw


env = Environment()
action_size = 9
state_size = 9

agent = TicTacToeAgent(state_size, action_size)
#agent = QAgent(state_size, action_size)

train_episodes = 5000
loss = 0

wins = 0
losses = 0
draws = 0
illegals = 0
moves = 0

for episode in range(train_episodes):

    state = env.reset()
    state = np.reshape(state, [1, state_size])
    # print("----------------------------------------------------------------")
    # print("New Game")
    # print("----------------------------------------------------------------")
    while True:
        # print("------------------------------------")
        # draw(state.reshape(9))
        action = agent.act(state)
        # print(action)
        moves += 1

        next_state, reward, done = env.step(action, train=True)

        if not (0 in next_state) and not done:
            print("ERROR Train")

        next_state = np.reshape(next_state, [1, state_size])

        if not (0 in state) and not done:
            print("ERROR Train")

        agent.remember(state.copy(), action, reward, next_state.copy(), done)

        # agent.print_memory()

        state = next_state

        if reward == env.REWARD_ILLEGAL:
            illegals += 1

        if done:
            if reward == env.REWARD_WIN:
                wins += 1
            elif reward == env.REWARD_LOSS or reward == env.REWARD_ILLEGAL:
                losses += 1
            elif reward == env.REWARD_DRAW:
                draws += 1
            break

    # print("------------------------------------")
    # draw(state.reshape(9))

    current_loss = agent.update()[0]
    loss += current_loss

    agent.write_loss_to_tensorboard(current_loss, episode)
    if episode % 100 == 0 and episode != 0:

        print("Episode: " + str(episode) + "/" + str(train_episodes) + "-> Test score = " + str(wins) + ", Loss : " + str(loss / 100))
        print("X: " + str(wins) + ", O: " + str(losses) + ", Draw: " + str(draws))
        print("Illegal: " + str(illegals))
        print("Moves: " + str(moves))
        loss = 0
        agent.write_score_to_tensorboard(wins, episode)
        wins = 0
        losses = 0
        draws = 0
        illegals = 0
        moves = 0



from TicTacToe.Environment import Environment
from TicTacToe.TicTacToeAgent import TicTacToeAgent
from QAgent import QAgent
import numpy as np
from TicTacToe.Environment import draw

def run_test_games(env, agent, nb_games):
    wins = 0
    losses = 0
    draws = 0
    illegals = 0
    moves = 0

    for episode in range(nb_games):

        state = env.reset()
        state = np.reshape(state, [1, state_size])
        while True:

            action = agent.act(state, is_training=False)
            moves += 1

            next_state, reward, done = env.step(action, done_on_ill=True)

            next_state = np.reshape(next_state, [1, state_size])

            state = next_state.copy()

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

    return wins, losses, draws, illegals

env = Environment()
action_size = 9
state_size = 9

agent = TicTacToeAgent(state_size, action_size)

train_episodes = 1000
loss = 0

wins = 0
losses = 0
draws = 0
illegals = 0
moves = 0

for episode in range(train_episodes):

    state = env.reset()
    state = np.reshape(state, [1, state_size])
    while True:

        action = agent.act(state)
        moves += 1

        next_state, reward, done = env.step(action)

        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state.copy(), action, reward, next_state.copy(), done)

        state = next_state.copy()

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

    current_loss = agent.update()[0]
    loss += current_loss

    agent.write_loss_to_tensorboard(current_loss, episode)
    if episode % 100 == 0 and episode != 0:

        print("Episode: " + str(episode) + "/" + str(train_episodes) + "-> Test score = " + str(wins) + ", Loss : " + str(loss / 100))
        print("X: " + str(wins) + ", O: " + str(losses) + ", Draw: " + str(draws))
        print("Illegal: " + str(illegals))
        print("Moves: " + str(moves))
        test_wins, test_losses, test_draws, test_illegals = run_test_games(env=env, agent=agent, nb_games=100)
        print("Test-> X: " + str(test_wins) + ", O: " + str(test_losses) + ", Draw: " + str(test_draws))
        agent.write_score_to_tensorboard(wins, episode)

        wins = 0
        losses = 0
        draws = 0
        illegals = 0
        moves = 0
        loss = 0

agent.save_model()

from TicTacToe.Environment import Environment
from TicTacToe.TicTacToeAgent import TicTacToeAgent
from RandomAgent import RandomAgent
from TicTacToe.Environment import get_first_turn, get_new_board
import numpy as np

env = Environment()
action_size = 9
state_size = 9

# agent = RandomAgent(action_size=9)
agent = TicTacToeAgent(state_size=state_size, action_size=action_size)
agent.load_model(filepath='/home/local/USHERBROOKE/judt3001/Bureau/Test/RL_QLearning_V2/TicTacToe/models/TicTacToe_2018-09-24-17-06.h5')

player = env.O
turn = get_first_turn()
done = False
state = get_new_board()
winner = env.REWARD_DEFAULT

while not done:
    print("------------------------")
    env.draw_board(numbers=True)
    if turn == env.X:
        state = np.reshape(state, [1, state_size])
        action = agent.act(state, is_training=False)
    else:
        action = int(input("Position?"))
    state, winner, done = env.step_player(action, turn)

    if winner != env.REWARD_ILLEGAL:
        turn = turn * -1
    else:
        if turn == player:
            print("Position is occupied: Enter new position")

    if done:
        break

print("------------------------")
env.draw_board()
if winner == player:
    print("Player win")
elif winner == env.DRAW:
    print("Draw")
else:
    print("Computer win")
import gym
from QAgent import QAgent
import numpy as np
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def run_tests(n, agent, env):
    score = 0
    for i in range(n):
        state = env.reset()
        state = np.array(np.eye(16)[state])
        state = np.reshape(state, [1, state_size])

        while True:

            action = agent.act(state, is_training=False)

            next_state, reward, done, _ = env.step(action)
            next_state = np.array(np.eye(16)[next_state])
            next_state = np.reshape(next_state, [1, state_size])

            state = next_state

            if done:
                score += reward
                break

    return score


env = gym.make('FrozenLake-v0')

action_size = env.action_space.n
state_size = 16

agent = QAgent(state_size, action_size)


Episodes = 2000
loss = 0

for episode in range(Episodes):

    state = env.reset()
    state = np.array(np.eye(16)[state])
    state = np.reshape(state, [1, state_size])

    for i in range(1000):

        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)
        next_state = np.array(np.eye(16)[next_state])
        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)

        state = next_state

        if done:
            break

    loss += agent.update()[0]

    if episode % 100 == 0 and episode != 0:
        score = run_tests(100, agent, env)
        print("Episode: " + str(episode) + "-> Test score = " + str(score) + ", Loss : " + str(loss/100))
        loss = 0


score = run_tests(1000, agent, env)
print("Finale test score = " + str(score))




import os

import gym
import numpy as np

from PolicyGradientAgent import PolicyGradientAgent

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


env = gym.make('LunarLander-v2')
action_size = env.action_space.n
state_size = 8

agent = PolicyGradientAgent(state_size, action_size)

print("Training...")
train_episodes = 5000
avg_score = 0
loss = 0
for episode in range(train_episodes):

    state = env.reset()
    state = np.reshape(state, [1, state_size])
    cum_reward = 0
    for i in range(1000):

        action = agent.act(state, is_training=True)

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward)

        state = next_state
        cum_reward += reward

        if done:
            avg_score += cum_reward
            break

    current_loss = agent.update()[0]
    loss += current_loss

    if episode % 100 == 0 and episode != 0:
        print("Episode: " + str(episode) + "/" + str(train_episodes) + ", score: " + str(avg_score/100) + ", Loss : " + str(loss/100))
        avg_score = 0
        loss = 0


print("Testing...")

test_episodes = 100
score = 0
for i in range(test_episodes):

    state = env.reset()
    state = np.reshape(state, [1, state_size])
    time = 0
    while True:

        action = agent.act(state, is_training=True)

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        env.render()

        state = next_state
        time += reward
        if done:
            score += time
            print("Test {}: score={}".format(i, time))
            break


print("Average test score: " + str(score / test_episodes))


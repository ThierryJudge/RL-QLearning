import os

import gym
import numpy as np

from Cartpole.CartpoleAgent import CartpoleAgent

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


env = gym.make('CartPole-v0')

action_size = env.action_space.n
state_size = 4


agent = CartpoleAgent(state_size, action_size)


print("Training...")
train_episodes = 20000
avg_score = 0
loss = 0
for episode in range(train_episodes):

    state = env.reset()
    state = np.reshape(state, [1, state_size])

    for i in range(1000):

        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)

        state = next_state

        if done:
            avg_score += i
            break

    current_loss = agent.update()[0]
    loss += current_loss

    agent.write_loss_to_tensorboard(current_loss, episode)
    agent.write_score_to_tensorboard(i, episode)

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

        action = agent.act(state, is_training=False)

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        state = next_state
        time += reward
        if done:
            score += time
            break


print("Average test score: " + str(score / test_episodes))
agent.write_value_to_tensorboard(score / test_episodes, 'Final Test Score', 0)

agent.save_model()

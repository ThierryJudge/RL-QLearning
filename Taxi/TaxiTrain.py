import gym
from Taxi.TaxiAgent import TaxiAgent
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def run_tests(n, agent, env):
    score = 0

    for i in range(n):
        state = env.reset()
        state = np.array(np.eye(state_size)[state])
        state = np.reshape(state, [1, state_size])
        episode_reward = 0
        while True:

            action = agent.act(state, is_training=False)
            #print(action)

            next_state, reward, done, _ = env.step(action)
            next_state = np.array(np.eye(state_size)[next_state])
            next_state = np.reshape(next_state, [1, state_size])

            state = next_state
            episode_reward += reward

            if done:
                print("Episode reward: " + str(episode_reward))
                score += episode_reward
                break

    return score/n


env = gym.make("Taxi-v2")
print(env)

action_size = env.action_space.n
state_size = env.observation_space.n

print(action_size)
print(state_size)
print(env.reset())

agent = TaxiAgent(state_size, action_size)


Episodes = 2000
loss = 0

for episode in range(Episodes):

    state = env.reset()
    state = np.array(np.eye(state_size)[state])
    state = np.reshape(state, [1, state_size])
    episode_reward = 0
    for i in range(1000):

        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)
        next_state = np.array(np.eye(state_size)[next_state])
        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            break

    current_loss = agent.update()[0]
    loss += current_loss
    agent.write_loss_to_tensorboard(current_loss, episode)


    if episode % 100 == 0 and episode != 0:
        score = run_tests(100, agent, env)
        print("Episode: " + str(episode) + "/" + str(Episodes) + "-> Test score = " + str(score) + ", Loss : " + str(loss/100))
        print("Last episode reward: " + str(episode_reward))
        agent.write_score_to_tensorboard(score, episode)
        loss = 0


score = run_tests(1000, agent, env)
print("Finale test score = " + str(score))
agent.write_value_to_tensorboard(score, 'Final Test Score', 0)





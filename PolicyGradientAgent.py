import numpy as np
from keras import backend as K, optimizers
from keras import utils as np_utils
from keras.layers import Dense
from keras.models import Sequential


class PolicyGradientAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.99
        self.learning_rate = 0.03
        self.learning_rate_decay = 0.01

        self.model = self._build_model()
        self.train_fn = self._get_train_fn()

        self.states = []
        self.actions = []
        self.rewards = []
        self.game_states = []
        self.game_actions = []
        self.game_rewards = []

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, input_dim=64, activation='relu'))
        model.add(Dense(self.action_size, input_dim=64, activation='softmax'))
        return model

    def _get_train_fn(self):

        action_prob_placeholder = self.model.output  # (N, A) get log probabilities from model (logits)
        action_onehot_placeholder = K.placeholder(shape=[None, self.action_size],
                                                  name='onehot')  # (N, A) get taken actions
        discount_reward_placeholder = K.placeholder(shape=[None], name='rewards')  # (N, )

        action_prob = K.sum((action_prob_placeholder * action_onehot_placeholder), axis=1)  # (N, )

        log_probs = K.log(action_prob)  # (N, )

        loss = K.mean(-log_probs * discount_reward_placeholder)

        optimizer = optimizers.Adam()

        updates = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)

        return K.function(inputs=[self.model.input,
                                  action_onehot_placeholder,
                                  discount_reward_placeholder],
                          outputs=[loss],
                          updates=updates)

    def update(self):
        states = np.squeeze(np.array(self.states))
        actions = np_utils.to_categorical(self.actions, num_classes=self.action_size)
        rewards = self.discount_rewards(np.array(self.rewards))

        loss = self.train_fn([states, actions, rewards])
        self.states = []
        self.actions = []
        self.rewards = []
        return loss

    def act(self, state, is_training=False):
        action_prob = np.squeeze(self.model.predict(state))
        if is_training:
            return np.random.choice(np.arange(self.action_size), p=action_prob)
        else:
            return np.argmax(action_prob)

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def remember_multiple(self, state, action, reward, done):
        self.game_states.append(state)
        self.game_actions.append(action)
        self.game_rewards.append(reward)

        if done:
            rewards = self.discount_rewards(np.array(self.game_rewards)).tolist()
            self.states.append(self.game_states)
            self.actions.append(self.game_actions)
            self.rewards.append(rewards)

            self.game_states = []
            self.game_actions = []
            self.game_rewards = []

    def update_multiple(self):
        states = np.squeeze(np.array(self.states))
        actions = np_utils.to_categorical(self.actions, num_classes=self.action_size)
        rewards = np.array(self.rewards)

        loss = self.train_fn([states, actions, rewards])
        self.states = []
        self.actions = []
        self.rewards = []
        return loss

    def discount_rewards(self, rewards):
        d = np.zeros_like(rewards)
        # baseline = np.mean(rewards)
        sum = 0
        for i in reversed(range(len(rewards))):
            sum = sum * self.gamma + rewards[i]
            d[i] = sum  # - baseline

        return d

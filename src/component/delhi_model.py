
import numpy as np
from collections import defaultdict


class TDQ:

    def __init__(self, num_states, num_actions, epsilon=0.2, alpha=0.1, gamma=0.9, seed=123):
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        np.random.seed(seed)
        self.Q = defaultdict(lambda: np.zeros(num_actions))

    def esoft(self, state):
        greedy_action = np.argmax(self.Q[state])
        action_probabilities = np.full(self.num_actions, self.epsilon / self.num_actions)
        action_probabilities[greedy_action] += (1.0 - self.epsilon)
        return np.random.choice(np.arange(self.num_actions), p=action_probabilities)

    def update(self, action,state, next_state,reward):
        max_next_q = np.max(self.Q[next_state])
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

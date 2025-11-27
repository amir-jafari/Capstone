import numpy as np

class TDQ:
  def __init__(self, num_states, num_actions, epsilon=0.2, alpha=0.1, gamma=0.9, seed=123):
    np.random.seed(seed)
    self.num_states = num_states
    self.num_actions = num_actions
    self.epsilon = epsilon
    self.alpha = alpha
    self.gamma = gamma
    self.Q = {(tuple([s]), a): 0.0 for s in range(num_states) for a in range(num_actions)}
    self.visit_counts = {}

  def get_q(self, state, action):
    key = (tuple(np.round(state, 4).flatten()), action)
    return self.Q.get(key, 0.0)

  def esoft(self, state, num_actions=None, return_probabilities=False):
    q_vals = [self.get_q(state, a) for a in range(num_actions)]
    policy = np.ones(num_actions) * self.epsilon / num_actions
    best_action = int(np.argmax(q_vals))
    policy[best_action] += (1.0 - self.epsilon)
    if return_probabilities:
        return policy
    return np.random.choice(np.arange(num_actions), p=policy)

  def update(self, action, state, next_state, reward):
    state_key = tuple(np.round(state, 4).flatten())
    old_q = self.get_q(state, action)
    next_q_max = max([self.get_q(next_state, a) for a in range(self.num_actions)])
    td_target = reward + self.gamma * next_q_max
    td_error = td_target - old_q
    # self.visit_counts[(state_key, action)] = self.visit_counts.get((state_key, action), 0) + 1
    # dynamic_alpha = 1.0 / (1 + self.visit_counts[(state_key, action)])  
    self.Q[(state_key, action)] = (1-self.alpha) * old_q + self.alpha * td_error


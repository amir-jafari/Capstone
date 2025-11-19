# --------------------------
# UCB
# --------------------------
import numpy as np
import pandas as pd
class UCB:
    def __init__(self, n_arms, manual_prior=None, random_seed=None):
        np.random.seed(random_seed)
        self.n_arms = n_arms
        if manual_prior:
            self.Q = np.ones((n_arms, 1)) * manual_prior[0]
            self.N = np.ones((n_arms, 1)) * manual_prior[1]
        else:
            self.Q = np.zeros((n_arms, 1))
            self.N = np.zeros((n_arms, 1))
        self.T = 0
        self._reset_logs()

    def _reset_logs(self):
        self.rewards_list = []
        self.chosen_arms = []
        self.rewards_matrix = np.zeros((0, self.n_arms))
        self.cumulative_regret = []

    def bandit(self, data, A, t):
        r_vec = data[t].copy()
        R = int(r_vec[A])
        row = np.zeros_like(r_vec); row[A] = R
        self.rewards_matrix = np.vstack([self.rewards_matrix, row.reshape(1, -1)])
        self.rewards_list.append(R)
        return R, r_vec

    def action(self):
        self.T += 1
        # Pull each arm once
        for a in range(self.n_arms):
            if self.N[a] == 0:
                return a
        bonus = np.sqrt((2.0 * np.log(self.T)) / self.N.flatten())
        ucb = self.Q.flatten() + bonus
        return int(np.argmax(ucb))

    def update(self, A, R):
        self.N[A] += 1
        self.Q[A] += (1.0 / self.N[A]) * (R - self.Q[A])

    def train(self, data, means_t=None):
        self._reset_logs()
        self.T = 0
        cum_regret = 0.0

        T = data.shape[0]
        for t in range(T):
            A = self.action()
            self.chosen_arms.append(A)

            R, r_vec = self.bandit(data, A, t)
            self.update(A, R)

            regret_t = float(np.max(r_vec) - R)
            cum_regret += regret_t
            self.cumulative_regret.append(cum_regret)

        table = self.create_table()
        return table, self.rewards_list, self.rewards_matrix, self.cumulative_regret

    def create_table(self):
        table = np.hstack([
            np.arange(1, self.n_arms + 1).reshape(self.n_arms, 1),
            self.N,
            self.Q.round(3)
        ]).astype(float)
        df = pd.DataFrame(table, columns=["Arm", "Count", "E[reward|arm]"])
        return df.to_string(index=False)

    def save(self):
        return pd.DataFrame({"N": self.N.flatten(), "Q": self.Q.flatten()})

    def recommend(self):
        best = int(np.argmax(self.Q))
        return best, float(self.Q[best])

    def reset(self):
        self.Q[:] = 0.0
        self.N[:] = 0.0
        self.T = 0
        self._reset_logs()
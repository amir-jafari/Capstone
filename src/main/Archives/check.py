#%%
import numpy as np
import matplotlib.pyplot as plt

def generate_drifting_env(n_observations=1000, n_arms=2, drift_scale=0.01, seed=None):
    """
    Generates a drifting (non-stationary) MAB environment where arm probabilities sum to 1.
    
    Parameters
    ----------
    n_observations : int
        Number of time steps.
    n_arms : int
        Number of arms.
    drift_scale : float
        Standard deviation of Gaussian noise controlling drift per step.
    seed : int or None
        Random seed for reproducibility.
        
    Returns
    -------
    env : np.ndarray
        Array of shape (n_observations, n_arms) containing arm probabilities over time.
    """
    rng = np.random.default_rng(seed)

    # Initialize equally (sum to 1)
    probs = np.ones(n_arms) / n_arms  
    env = np.zeros((n_observations, n_arms))
    env[0] = probs

    for t in range(1, n_observations):
        # Add Gaussian drift
        probs += rng.normal(0, drift_scale, size=n_arms)
        # Keep positive and normalize so sum = 1
        probs = np.clip(probs, 1e-6, None)
        probs /= probs.sum()
        env[t] = probs

    return env

# Example usage
n_obs = 500
n_arms = 2

env = generate_drifting_env(n_obs, n_arms, drift_scale=0.02, seed=100)

# Plot the evolution of probabilities
plt.figure(figsize=(10, 5))
for i in range(n_arms):
    plt.plot(env[:, i], label=f"Arm {i+1}")
plt.title("Drifting Arm Probabilities (Sum = 1)")
plt.xlabel("Time step")
plt.ylabel("Arm Probability")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# %%

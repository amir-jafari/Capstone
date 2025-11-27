#%%
# -*- coding: utf-8 -*-
"""
Author: Hema Chandra Puchakayala
Date: 2025-09-24
Version: 1.0 (with UCB integration)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

project_root = "/Users/hema/Desktop/GWU/Aug_2025/Capstone/fall-2025-group12"
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.component.env_downscale import *
from src.component.model_ucb import UCB

# Hyperparameters
NUM_USERS = 3
NUM_DOCUMENTS = 4
SEED_DOC = 64
SEED = 1234
CATEGORIES = ['Sports', 'Politics']

ALPHA = 0.2  
A = 2         
B = 5         
BETA1 = 0.2  
BETA2 = 0.2   
GAMMA1 = 0.3
GAMMA2 = 0.3

#%%
env = DriftingEnvironment(NUM_USERS, NUM_DOCUMENTS, ALPHA, A, B, BETA1, BETA2, GAMMA1, GAMMA2, CATEGORIES, SEED, SEED_DOC)

# Model hyperparameters
EPISODES = 100
NUM_ROUNDS = 365
color = ['r', 'b']

all_random_rewards = np.zeros((EPISODES, NUM_ROUNDS))
all_ucb_rewards = np.zeros((EPISODES, NUM_ROUNDS))

all_latent_prefs_actual = []
all_latent_prefs_random = []  
all_latent_prefs_ucb = []

ucb_cum_rew_list = []
random_cum_rew_list = []

last_latent_pref_ucb = []
last_latent_pref_random = []

users, documents = env.reset()

#%%
for episode in range(EPISODES):
    print(f"\n--- Episode {episode + 1} ---")

    # Random baseline
    users, documents = env.reset()
    random_cum_rew = 0
    latent_preference_list_random = []

    for step in range(NUM_ROUNDS):
        for user_index, user in enumerate(users):
            action = np.random.choice(NUM_DOCUMENTS)
            selected_document = documents[action]
            reward, updated_user = env.step(user, selected_document)
            random_cum_rew += reward
            users[user_index] = updated_user
            latent_preference_list_random.append(updated_user.theta.copy())
            if episode == 0:
                all_latent_prefs_actual.append(updated_user.theta.copy())
            if episode == EPISODES - 1:
                last_latent_pref_random.append(updated_user.theta.copy())
    random_cum_rew_list.append(random_cum_rew)
    all_latent_prefs_random.append(np.array(latent_preference_list_random))

    # UCB baseline
    users, documents = env.reset()
    ucb_cum_rew = 0
    latent_preference_list_ucb = []

    model = UCB(n_arms=NUM_DOCUMENTS, random_seed=SEED + episode)

    for step in range(NUM_ROUNDS):
        for user_index, user in enumerate(users):
            action = model.action()
            selected_document = documents[action]
            reward, updated_user = env.step(user, selected_document)
            ucb_cum_rew += reward
            users[user_index] = updated_user

            scalar_reward = reward if np.isscalar(reward) else np.sum(reward)
            latent_preference_list_ucb.append(updated_user.theta.copy())
            model.update(action, scalar_reward)
            if episode == EPISODES - 1:
                last_latent_pref_ucb.append(updated_user.theta.copy())
    ucb_cum_rew_list.append(ucb_cum_rew)
    all_latent_prefs_ucb.append(np.array(latent_preference_list_ucb))

    if episode % 10 == 0:
        avg_latent_prefs_random = np.mean(np.array(all_latent_prefs_random), axis=0)
        avg_latent_prefs_ucb = np.mean(np.array(all_latent_prefs_ucb), axis=0)
        num_categories = avg_latent_prefs_ucb.shape[1]

        for category_index in range(num_categories):
            plt.plot(avg_latent_prefs_random[:, category_index], linestyle='dashed', label=f'Random {CATEGORIES[category_index]}')
            plt.plot(avg_latent_prefs_ucb[:, category_index], label=f'UCB {CATEGORIES[category_index]}', color=color[category_index])

        plt.xlabel('Time (Rounds)')
        plt.ylabel('Average Latent Preference Score')
        plt.title(f'Average Latent Preferences Over Episodes {episode}: Random vs UCB')
        plt.legend(loc='upper right', fontsize='small', ncol=2)
        plt.show()

avg_latent_prefs_random = np.mean(np.array(all_latent_prefs_random), axis=0)
avg_latent_prefs_ucb = np.mean(np.array(all_latent_prefs_ucb), axis=0)

num_categories = avg_latent_prefs_ucb.shape[1]
last_latent_pref_random = np.array(last_latent_pref_random)
last_latent_pref_ucb = np.array(last_latent_pref_ucb)

for category_index in range(num_categories):
    plt.plot(last_latent_pref_random[:, category_index], linestyle='dashed', label=f'Random {CATEGORIES[category_index]}')
    plt.plot(last_latent_pref_ucb[:, category_index], label=f'UCB {CATEGORIES[category_index]}', color=color[category_index])

plt.xlabel('Time (Rounds)')
plt.ylabel('Last Latent Preference Score')
plt.title('Last Latent Preferences: Random vs UCB')
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(np.array(random_cum_rew_list), label='Random Model')
plt.plot(np.array(ucb_cum_rew_list), label='UCB Model')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward Per Episode')
plt.legend()
plt.grid(True)
plt.show()

#%% ---------------------------------
# Plot cumulative reward over all episodes (optional)
cumulative_random_rewards = np.cumsum(random_cum_rew_list)
cumulative_ucb_rewards = np.cumsum(ucb_cum_rew_list)

plt.figure(figsize=(10, 6))
plt.plot(cumulative_random_rewards, label='Random Model Cumulative Reward')
plt.plot(cumulative_ucb_rewards, label='UCB Model Cumulative Reward')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward Over Episodes')
plt.legend()
plt.grid(True)
plt.show()

#%% ---------------------------------
# Plot drift of actual latent preferences (for one user across all rounds/episodes)
color = ['r','b']
actual_latent_preferences = np.array(all_latent_prefs_actual)
plt.figure(figsize=(10, 6))
for category_index in range(actual_latent_preferences.shape[1]):
    plt.plot(actual_latent_preferences[:, category_index], label=f'{CATEGORIES[category_index]}', color=color[category_index])
plt.xlabel('Time (Rounds)')
plt.ylabel('Latent Preference Score')
plt.title('User Latent Preferences Drift Over Time')
plt.legend()
plt.show()

#%% ---------------------------------
# Average latent preference evolution (Random vs UCB)
num_categories = avg_latent_prefs_ucb.shape[1]
for category_index in range(num_categories):
    plt.plot(avg_latent_prefs_random[:, category_index], linestyle='dashed', label=f'Random {CATEGORIES[category_index]}')
    plt.plot(avg_latent_prefs_ucb[:, category_index], label=f'UCB {CATEGORIES[category_index]}', color=color[category_index])

plt.xlabel('Time (Rounds)')
plt.ylabel('Average Latent Preference Score')
plt.title('Average Latent Preferences Over Episodes: Random vs UCB')
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.show()

#%% ---------------------------------
# Document topic distributions bar plot
def add_labels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], round(y[i], 2))

plt.figure(figsize=(10, 5))
plt.bar(range(len(documents[0].x)), height=documents[0].x, label=f'{CATEGORIES[0]}', color='r')
add_labels(CATEGORIES, documents[0].x)
plt.xlabel('Categories')
plt.ylabel('Topic Distribution')
plt.title('Document 0 Topic Distribution')
plt.xticks(range(len(CATEGORIES)), CATEGORIES, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(range(len(documents[1].x)), height=documents[1].x, label=f'{CATEGORIES[1]}', color='b')
add_labels(CATEGORIES, documents[1].x)
plt.xlabel('Categories')
plt.ylabel('Topic Distribution')
plt.title('Document 1 Topic Distribution')
plt.xticks(range(len(CATEGORIES)), CATEGORIES, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()



# %%

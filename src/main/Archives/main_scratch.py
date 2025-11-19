#%%
import os
import sys
project_root = "/Users/hema/Desktop/GWU/Aug_2025/Capstone/fall-2025-group12"
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# sys.path.insert(0, '/Users/hema/Desktop/GWU/Aug_2025/Capstone/fall-2025-group12')
from src.component.env_scratch import * 
from src.component.model import *
import numpy as np
import matplotlib.pyplot as plt

# %%
'''
Environment Setup 
'''
CATEGORIES = ['Sports','Politics']
ALPHA_DOC = 0.2
A_DOC = 2
B_DOC = 5
SEED_DOC = 46
NUM_USERS = 1
NUM_DOCS = 2
BETA1 = 0.2
BETA2 = 0.2

SEED_RESPONSE = 1234

env = DriftingEnvironment(categories = CATEGORIES,alpha_doc=ALPHA_DOC,a_doc = A_DOC,b_doc = B_DOC,seed_doc = SEED_DOC, num_users = NUM_USERS,num_docs = NUM_DOCS,beta1 = BETA1, beta2 = BETA2, seed_response = SEED_RESPONSE)

users,documents = env.reset()

# %%
def add_labels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i])
plt.figure(figsize=(10, 5))
plt.bar(range(len(documents[0].x)), height=documents[0].x, label=f'{CATEGORIES[1]}',color = 'r')
add_labels(CATEGORIES, documents[0].x)
plt.xlabel('Categories')
plt.ylabel('Topic Distribution')
plt.title('Document Topic Distributions')
plt.xticks(range(len(CATEGORIES)), CATEGORIES, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
# %%
def add_labels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i])
plt.figure(figsize=(10, 5))
plt.bar(range(len(documents[1].x)), height=documents[1].x, label=f'{CATEGORIES[0]}',color = 'b')
add_labels(CATEGORIES, documents[1].x)
plt.xlabel('Categories')
plt.ylabel('Topic Distribution')
plt.title('Document Topic Distributions')
plt.xticks(range(len(CATEGORIES)), CATEGORIES, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# %%
'''
Random Model:
'''

EPISODES = 100
ROUNDS = 365

user_normal_drift_r = [] # size : (ROUNDS,NUM_CATEGORIES) * NUM_USERS
all_latent_pref_random = []

for episode in range(EPISODES):
    print(f'--Episode {episode+1}:--')
    users,_ = env.reset()
    cum_reward = 0
    latent_pref_random = []

    for step in range(ROUNDS):
        for user_index, user in enumerate(users):
            if episode == 0 and step == 0:
                user_normal_drift_r.append(user.theta.copy())
            action = np.random.choice(NUM_DOCS)
            selected_document = documents[action]
            reward, updated_user = env.step(user, selected_document)
            cum_reward += reward
            users[user_index] = updated_user
            latent_pref_random.append(updated_user.theta.copy())
            if episode == 0:
                user_normal_drift_r.append(updated_user.theta.copy())
            
    all_latent_pref_random.append(np.array(latent_pref_random))

    # users,_ = env.reset()
    # cum_reward = 0
    # latent_pref_q = []

    # for step in range(ROUNDS):
    #     for user_index, user in enumerate(users):
    #         if episode == 0:
    #             user_normal_drift_q.append(user.theta)


#%%
color = ['r','b']
actual_latent_preferences = np.array(user_normal_drift_r) 
print(actual_latent_preferences.sum(axis=1)) 
plt.figure(figsize=(10, 6))
for category_index in range(actual_latent_preferences.shape[1]):
    plt.plot(actual_latent_preferences[:, category_index], label=f'{CATEGORIES[category_index]}',color = color[category_index])

plt.xlabel('Time (Rounds)')
plt.ylabel('Latent Preference Score')
plt.title('User Latent Preferences Drift Over Time')
plt.legend()
plt.show()


            


# %%

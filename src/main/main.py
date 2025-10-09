#%%
# -*- coding: utf-8 -*-
"""
Author: Hema Chandra Puchakayala
Date: 2025-09-24
Version: 1.0
"""


import os
import sys
# project_root = "/Users/hema/Desktop/GWU/Aug_2025/Capstone/fall-2025-group12"
os.chdir('/../..')
# sys.path.insert(0, '/Users/hema/Desktop/GWU/Aug_2025/Capstone/fall-2025-group12')
from src.component.environment import * 

print('Generating Documents')
print('-'*50)
sampler = DriftingDocumentSampler(categories, alpha=0.05, a=2, b=5 )
docs = sampler.sample_documents(100)

for d in docs:
  print(d.create_observation())

print('='*50)

print('Generating Users')
print('-'*50)
sampler = DriftingUserSampler(categories, alpha=0.05, a=2, b=5 )
users = sampler.sample_users(100)

for u in users:
  print(u.create_observation())

print('='*50)
# %%

# -*- coding: utf-8 -*-
"""
Author: Hema Chandra Puchakayala
Date: 2025-09-24
Version: 1.0
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install ipynb

!pip install --upgrade --no-cache-dir recsim

from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir("/content/drive/MyDrive/Colab Notebooks")

import ipynb.fs.full.Environment as env

sampler = env.DriftingDocumentSampler(env.categories, alpha=0.05, a=2, b=5 )
docs = sampler.sample_documents(100)

for d in docs:
  print(d.create_observation())


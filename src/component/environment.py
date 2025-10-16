# -*- coding: utf-8 -*-
"""
Author: Hema Chandra Puchakayala
Date: 2025-09-24
Version: 1.0
"""


import numpy as np
import gymnasium as gym
import recsim.document
import recsim.user

class DriftingDocument(recsim.document.AbstractDocument):
  """
  This class would hold documents with features:
  topic,
  popularity score, and
  quality score.
  """
  def __init__(self, doc_id, x, p, q, num_categories):
    super(DriftingDocument, self).__init__(doc_id)
    self.x = x   # topic
    self.p = p   # popularity
    self.q = q   # quality
    self.num_categories = num_categories

  def create_observation(self):
    return {
        "topic": self.x,
        "popularity": self.p,
        "quality": self.q
    }

  def observation_space(self):
    return gym.spaces.Dict({
        "topic": gym.spaces.Discrete(self.num_categories),
        "popularity": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
        "quality": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)
    })

class DriftingDocumentSampler(recsim.document.AbstractDocumentSampler):
  """
  This class would generate documents at each time step by sampling their features from a distribution.
  """
  def __init__(self, CATEGORIES, alpha=0.1, a=2, b=5):
    self.CATEGORIES = CATEGORIES
    self.num_categories = len(CATEGORIES)
    self.cat2id = {c: i for i, c in enumerate(CATEGORIES)}
    self.alpha = np.full(self.num_categories, alpha)
    # self.category_probs = np.random.dirichlet(self.alpha)
    self.a = a
    self.b = b

  def sample_document(self, doc_id):
    x = np.random.dirichlet(self.alpha)
    p = np.random.beta(self.a, self.b)
    q = np.random.beta(self.b, self.a)

    return DriftingDocument(doc_id, x, p, q, self.num_categories)


  def sample_documents(self, num_docs):
    return [self.sample_document(doc_id) for doc_id in range(num_docs)]

class DriftingUserState(recsim.user.AbstractUserState):
  '''
  This class would store for a particular user at time: t, latent preference vector: theta, user satisfaction or engagement score: sigma, fatigue variable :phi, and current timestep: tau.
  '''
  def __init__(self, user_id, theta, sigma, phi, num_categories, tau=0):
    self.theta = theta
    self.sigma = sigma
    self.phi = phi
    self.tau = tau
    self.num_categories = num_categories

  def create_observation(self):
    return {
        "theta": self.theta,
        "sigma": self.sigma,
        "phi": self.phi #,
        # "tau": self.tau
    }

  def observation_space(self):
    return gym.spaces.Dict({
        "theta": gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_categories,), dtype=np.float32),
        "sigma": gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_categories,), dtype=np.float32),
        "phi": gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_categories,), dtype=np.float32),
        # "tau": gym.spaces.Discrete(self.tau)
    })

class DriftingUserSampler(recsim.user.AbstractUserSampler):
  '''
  This class would generate a population of users at time t = 0:
  '''
  def __init__(self, categories, alpha=0.1, a=2, b=5):
    self.categories = categories
    self.num_categories = len(categories)
    self.alpha = np.full(self.num_categories, alpha)
    self.category_probs = np.random.dirichlet(self.alpha)
    self.a = a
    self.b = b 

  def sample_user(self, user_id):
    theta = np.ones(self.num_categories) / self.num_categories
    sigma = np.random.beta(self.a, self.b)
    phi = np.random.beta(self.b, self.a)                          
    return DriftingUserState(user_id, theta, sigma, phi, self.num_categories)

  def sample_users(self, num_users):
    return [self.sample_user(user_id) for user_id in range(num_users)]


class DriftingResponseModel(recsim.user.AbstractResponse):
  """
  This class defines how a user reacts to a document, mapping (user state, document features) into probabilities of click.
  """
  def __init__(self, beta1=None, beta2=None):
    self.beta1 = beta1
    self.beta2 = beta2
  
  def sigmoid(self,x):
    return 1.0 / (1.0 + np.exp(-x))

  def score(self, user_state, doc):
    epsilon = np.random.normal(0, 0.05)
    theta_x = np.dot(user_state.theta, doc.x)
    score = theta_x + self.beta1 * doc.q + self.beta2 * doc.p + epsilon
    prob_click = self.sigmoid(score)
    return prob_click

  def simulate_response(self, user_state, doc):
    click_prob = self.score(user_state, doc)
    click = np.random.binomial(1, click_prob)
    return click
  
  @staticmethod
  def response_space():
    """ArraySpec that defines how a single response is represented."""
    return gym.spaces.Discrete(2)  # 0: no-click, 1: click

  def create_observation(self):
    """Creates a tensor observation of this response."""
    return {'beta1': self.beta1, 'beta2': self.beta2}

class DriftingUserModel(recsim.user.AbstractUserModel):
  """
  This class would coordinate the simulation: advancing the user state, calling the response model, and updating preferences via a drift process.
  """
  def __init__(self, alpha=None):
    self.alpha = alpha
    self.response_model = DriftingResponseModel(beta1=0.3, beta2=0.3) 

  def update_state(self, user_state, doc):
    epsilon = np.random.normal(0, 0.005, size=user_state.theta.shape)
    user_state.theta = (1.0 - self.alpha) * user_state.theta + self.alpha * doc.x + epsilon
    user_state.theta = user_state.theta / np.sum(user_state.theta)  # Normalize to sum to 1
    
    # user_state.tau += 1  
    return user_state
  
  def is_terminal(self):
    # Return True/False to indicate if user session should terminate
    # For now, always return False
    return False
  
  def simulate_response(self, documents,userstate):
    # Simulate user's responses to a slate of documents
    return [self.response_model.simulate_response(userstate, doc) for doc in documents]
  
  # def update_state(self, slate_documents, responses):
  #   # Update user's internal state based on recommended slate and their response
  #   # updated_latent_pref = (1-self.alpha) * user_latent_vector + self.alpha * doc_topic_vector + epsilon
  #   pass

class DriftingEnvironment:
  """
  This class would combine the document sampler, user sampler, user model, and response model to create the full environment.
  """
  def __init__(self,num_users,num_documents,alpha,a,b,beta_1,beta_2, categories,seed):
    np.random.seed(seed)
    self.num_users = num_users
    self.num_documents = num_documents
    self.categories = categories
    self.alpha = alpha
    self.a = a
    self.b = b
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.doc_sampler = DriftingDocumentSampler(self.categories,alpha=0.1, a=2, b=5)
    self.user_sampler = DriftingUserSampler(self.categories,alpha=0.1, a=2, b=5)
    self.response_model = DriftingResponseModel(beta1=0.3, beta2=0.3)
    self.user_model = DriftingUserModel(alpha=0.01)
  
  def step(self, user, doc):
    response = self.response_model.simulate_response(user, doc)
    user = self.user_model.update_state(user, doc)
    return response, user
  
  def reset(self):
    self.users = self.user_sampler.sample_users(self.num_users)
    self.documents = self.doc_sampler.sample_documents(self.num_documents)
    return self.users, self.documents

    

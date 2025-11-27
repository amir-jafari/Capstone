import numpy as np
import gymnasium as gym
import recsim.document
import recsim.user

class DriftingDocument(recsim.document.AbstractDocument):
    """This class holds documents with features: topic, popularity, and quality."""
    def __init__(self, doc_id, x, p, q, num_categories):
        super(DriftingDocument, self).__init__(doc_id)
        self.x = x   # topic
        self.p = p   # popularity
        self.q = q   # quality
        self.num_categories = num_categories

    def create_observation(self):
        return {"topic": self.x, "popularity": self.p, "quality": self.q}

    def observation_space(self):
        return gym.spaces.Dict({
            "topic": gym.spaces.Discrete(self.num_categories),
            "popularity": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            "quality": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)
        })


class DriftingDocumentSampler(recsim.document.AbstractDocumentSampler):
    """Generates documents by sampling their features from distributions."""
    def __init__(self, CATEGORIES, alpha=0.1, a=2, b=5, seed=None):
        self.CATEGORIES = CATEGORIES
        self.num_categories = len(CATEGORIES)
        self.cat2id = {c: i for i, c in enumerate(CATEGORIES)}
        # self.alpha = np.full(self.num_categories, alpha)
        self.alpha = alpha

        self.a = a
        self.b = b
        self.seed = seed
        # self.rng = np.random.default_rng(seed)
        self.min_val = 0

    def sample_document(self, doc_id):
        rng = np.random.default_rng(self.seed + (doc_id+1) * self.num_categories)
        x = rng.dirichlet([(doc_id+1) * self.alpha] * self.num_categories)
        p = rng.beta((doc_id+1) * self.a,(doc_id+1) * self.b)
        q = rng.beta((doc_id+1) * self.b,(doc_id+1) * self.a)
        return DriftingDocument(doc_id, x, p, q, self.num_categories)

    def sample_documents(self, num_docs):
        return [self.sample_document(doc_id) for doc_id in range(num_docs)]


class DriftingUserState(recsim.user.AbstractUserState):
    """Represents a user's latent preference state at time t."""
    def __init__(self, user_id, theta, num_categories):
        self.theta = theta
        self.num_categories = num_categories

    def create_observation(self):
        return {"theta": self.theta}

    def observation_space(self):
        return gym.spaces.Dict({
            "theta": gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_categories,), dtype=np.float32)
        })


class DriftingUserSampler(recsim.user.AbstractUserSampler):
    """Generates a population of users at t = 0."""
    def __init__(self, categories, alpha=0.1, a=2, b=5, seed=None):
        self.categories = categories
        self.num_categories = len(categories)
        self.alpha = np.full(self.num_categories, alpha)
        self.rng = np.random.default_rng(seed)
        self.category_probs = self.rng.dirichlet(self.alpha)
        self.a = a
        self.b = b

    def sample_user(self, user_id):
        theta = np.ones(self.num_categories) / self.num_categories
        
        return DriftingUserState(user_id, theta, self.num_categories)

    def sample_users(self, num_users):
        return [self.sample_user(user_id) for user_id in range(num_users)]


class DriftingResponseModel(recsim.user.AbstractResponse):
    """Defines how a user reacts to a document (probability of click)."""
    def __init__(self, beta1=None, beta2=None,gamma1 = None, gamma2 = None, seed=None):
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.rng = np.random.default_rng(seed)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def score(self, user_state, doc):
        epsilon = self.rng.normal(0, 0.05)
        theta_x = np.dot(user_state.theta, doc.x)
        score = theta_x + self.beta1 * doc.q + self.beta2 * doc.p + epsilon
        prob_click = self.sigmoid(score)
        return prob_click

    def simulate_response(self, user_state, doc):
        click_prob = self.score(user_state, doc)
        click = self.rng.binomial(1, click_prob)
        return click

    @staticmethod
    def response_space():
        return gym.spaces.Discrete(2)

    def create_observation(self):
        return {'beta1': self.beta1, 'beta2': self.beta2}


class DriftingUserModel(recsim.user.AbstractUserModel):
    """Coordinates simulation, updates user state via a drift process."""
    def __init__(self, alpha=None, seed=None):
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.response_model = DriftingResponseModel(beta1=0.3, beta2=0.3,gamma1 = 0.3,gamma2 = 0.3, seed=seed)

    def update_state(self, user_state, doc):
        epsilon = self.rng.normal(0, 0.005, size=user_state.theta.shape)
        epsilon_sigma = self.rng.normal(0, 0.05)
        epsilon_phi = self.rng.normal(0, 0.05)
        user_state.theta = (1.0 - self.alpha) * user_state.theta + self.alpha * doc.x + epsilon
        user_state.theta = user_state.theta / np.sum(user_state.theta)
        return user_state

    def is_terminal(self):
        return False

    def simulate_response(self, documents, user_state):
        return [self.response_model.simulate_response(user_state, doc) for doc in documents]


class DriftingEnvironment:
    """Combines all components into a full environment."""
    def __init__(self, num_users, num_documents, alpha, a, b, beta_1, beta_2, gamma1, gamma2, categories, seed=None, seed_doc = None):
        self.num_users = num_users
        self.num_documents = num_documents
        self.categories = categories
        self.alpha = alpha
        self.a = a
        self.b = b
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.seed_doc = seed_doc
        self.gamma1 = gamma1
        self.gamma2 = gamma2

        self.doc_sampler = DriftingDocumentSampler(categories, alpha=alpha, a=a, b=b, seed=seed_doc)
        self.user_sampler = DriftingUserSampler(categories, alpha=alpha, a=a, b=b, seed=seed)
        self.response_model = DriftingResponseModel(beta1=beta_1, beta2=beta_2, gamma1 = gamma1,gamma2=gamma2, seed=seed)
        self.user_model = DriftingUserModel(alpha=0.01, seed=seed)

    def step(self, user, doc):
        # self.response_model = DriftingResponseModel(beta1=self.beta_1, beta2=self.beta_2, gamma1 = self.gamma1,gamma2=self.gamma2, seed=self.seed)
        # self.user_model = DriftingUserModel(alpha=0.01, seed=self.seed)
        response = self.response_model.simulate_response(user, doc)
        user = self.user_model.update_state(user, doc)
        return int(response), user

    def reset(self):
        # self.doc_sampler = DriftingDocumentSampler(self.categories, alpha=self.alpha, a=self.a, b=self.b, seed=self.seed)
        # self.user_sampler = DriftingUserSampler(self.categories, alpha=self.alpha, a=self.a, b=self.b, seed=self.seed)
        users = self.user_sampler.sample_users(self.num_users)
        documents = self.doc_sampler.sample_documents(self.num_documents)
        return users, documents

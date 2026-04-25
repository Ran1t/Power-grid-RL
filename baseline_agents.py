"""
Baseline agents for PowerGrid microgrid evaluation.

Agents:
  RandomAgent      - uniformly random actions (lower bound)
  ConservativeAgent - do-nothing, never moves battery (passive baseline)
  QLearningAgent   - tabular Q-learning, trains then evaluates
  SARSAAgent       - tabular SARSA (on-policy TD), trains then evaluates

State space (discretized): hour-bin(8) x soc-bin(5) x price-bin(3) = 120 states
Action space: battery commands in {-1.0, -0.5, 0.0, 0.5, 1.0}
"""

import random
import numpy as np


class RandomAgent:
    name = "Random"

    def act(self, obs):
        return random.uniform(-1.0, 1.0), 0.0

    def update(self, reward, next_obs, done):
        pass


class ConservativeAgent:
    name = "Conservative"

    def act(self, obs):
        return 0.0, 0.0

    def update(self, reward, next_obs, done):
        pass


def _discretize(obs):
    h = min(7, int(obs.hour / 3))
    s = min(4, int(obs.battery_soc * 5))
    p = 0 if obs.price <= 0.09 else (1 if obs.price <= 0.16 else 2)
    return h, s, p


_ACTIONS = [-1.0, -0.5, 0.0, 0.5, 1.0]


class QLearningAgent:
    """Off-policy TD: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s') - Q(s,a)]"""
    name = "Q-Learning"

    def __init__(self, alpha=0.15, gamma=0.95, epsilon=0.20):
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon
        self.Q       = np.zeros((8, 5, 3, len(_ACTIONS)))
        self.training = True
        self._s = None
        self._a = None

    def act(self, obs):
        st = _discretize(obs)
        if self.training and np.random.random() < self.epsilon:
            ai = np.random.randint(len(_ACTIONS))
        else:
            ai = int(np.argmax(self.Q[st]))
        self._s, self._a = st, ai
        return _ACTIONS[ai], 0.0

    def update(self, reward, next_obs, done):
        if not self.training or self._s is None:
            return
        ns     = _discretize(next_obs)
        q_next = 0.0 if done else float(np.max(self.Q[ns]))
        self.Q[self._s][self._a] += self.alpha * (
            reward + self.gamma * q_next - self.Q[self._s][self._a]
        )

    def set_eval(self):
        self.training = False


class SARSAAgent:
    """On-policy TD: Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s', a') - Q(s,a)]"""
    name = "SARSA"

    def __init__(self, alpha=0.15, gamma=0.95, epsilon=0.20):
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon
        self.Q       = np.zeros((8, 5, 3, len(_ACTIONS)))
        self.training = True
        self._s = None
        self._a = None

    def act(self, obs):
        st = _discretize(obs)
        if self.training and np.random.random() < self.epsilon:
            ai = np.random.randint(len(_ACTIONS))
        else:
            ai = int(np.argmax(self.Q[st]))
        self._s, self._a = st, ai
        return _ACTIONS[ai], 0.0

    def update(self, reward, next_obs, done):
        if not self.training or self._s is None:
            return
        ns = _discretize(next_obs)
        # on-policy: sample next action with epsilon-greedy
        na = (np.random.randint(len(_ACTIONS))
              if np.random.random() < self.epsilon
              else int(np.argmax(self.Q[ns])))
        q_next = 0.0 if done else float(self.Q[ns][na])
        self.Q[self._s][self._a] += self.alpha * (
            reward + self.gamma * q_next - self.Q[self._s][self._a]
        )

    def set_eval(self):
        self.training = False


ALL_AGENTS = [RandomAgent, ConservativeAgent, QLearningAgent, SARSAAgent]

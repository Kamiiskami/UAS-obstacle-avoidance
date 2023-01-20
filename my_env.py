from gym.envs.toy_text.discrete import DiscreteEnv
import numpy as np
UP_RIGHT = 4
UP_LEFT = 5
DOWN_RIGHT = 6
DOWN_LEFT = 7
# from ObstacleAvoidanceENV import 

from gym import Env, spaces
from gym.utils import seeding


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

class MyEnv(DiscreteEnv):
    def __init__(self, nS, nA, P, isd, wind_direction, wind_strength):
        self.P = P
        self.isd = isd
        self.lastaction = None  # for rendering
        self.nS = nS
        self.nA = nA
        self.wind_direction = wind_direction
        self.wind_strength = wind_strength
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        self.seed()
        self.s = categorical_sample(self.isd, self.np_random)
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return int(self.s)

    def step(self, a, wind_direction, wind_strength):
        if a == UP_RIGHT:
            transitions = self.P[self.s][UP_RIGHT]
            i = categorical_sample([t[0] for t in transitions], self.np_random)
            prob, next_state, reward, done = str(transitions)[i]
            self.s = next_state
            self.lastaction = a
            self.wind_direction = wind_direction
            return (next_state, reward, done, prob, self.wind_direction, self.wind_strength)
        elif a == UP_LEFT:
            transitions = self.P[self.s][UP_LEFT]
            i = categorical_sample([t[0] for t in transitions], self.np_random)
            prob, next_state, reward, done = str(transitions)[i]
            self.s = next_state
            self.lastaction = a
            self.wind_direction = wind_direction
            return (next_state, reward, done,  prob,  self.wind_direction,  self.wind_strength)
        elif a == DOWN_RIGHT:
            transitions = self.P[self.s][DOWN_RIGHT]
            i = categorical_sample([t[0] for t in transitions], self.np_random)
            prob, next_state, reward, done = str(transitions)[i]
            self.s = next_state
            self.lastaction = a
            self.wind_direction = wind_direction
            return (next_state, reward, done,  prob,  self.wind_direction,  self.wind_strength)
        elif a == DOWN_LEFT:
            transitions = self.P[self.s][DOWN_LEFT]
            i = categorical_sample([t[0] for t in transitions], self.np_random)
            prob, next_state, reward, done = str(transitions)[i]
            self.s = next_state
            self.lastaction = a
            self.wind_direction = wind_direction
            return (next_state, reward, done,  prob,  self.wind_direction,  self.wind_strength)
        else:
            transitions = self.P[self.s][a]
            i = categorical_sample([t[0] for t in transitions], self.np_random)
            p, s, r, d = str(transitions)[i]
            self.s = s
            self.lastaction = a
            return (int(s), r, d,  p,  self.wind_direction,  self.wind_strength)
import gym
import numpy as np
from gym import spaces
from hypergraph import *
from reward import score
import itertools 
from stable_baselines3 import PPO, SAC,A2C, DQN
from stable_baselines3.common.env_checker import check_env

N = 8
state_dim = 2**N
list_N = range(1,N+1)
combs_N = []
for i in range(1,N+1):
	combs_N.extend(list(itertools.combinations(list_N,i)))

def calcScore(state):
  primal = []
  for (i,j) in zip(combs_N,range(state_dim)):
    if (state[j] == 1):
      primal.append(i)

  #primal = reduce(primal)
  return score(reduce(primal))




class HyperGraphEnv(gym.Env):


  def __init__(self):
    super(HyperGraphEnv, self).__init__()
    self.action_space = spaces.Discrete(2)
    self.observation_space = spaces.Box(shape=(2 * state_dim,), dtype=np.int32, low=0, high=1)
    
    self._state = np.zeros(2 * state_dim)
    self._episode_ended = False
    self.time = 0
    self.reward = 0
    self.mask = np.zeros(state_dim)


  def reset(self):
    self._state = np.zeros(state_dim * 2)
    self._episode_ended = False
    self.time = 0
    self.mask = np.zeros(state_dim)
    self.reward=0
    return np.array(self._state, dtype=np.int32)

  def step(self, action):
    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      self.reset()

    # Make sure episodes don't go on forever.
    if self.mask[self.time] == 1:
      self._state[self.time] = 0
    elif action == 1:
      #mask out all of the next actions
      for i in range(state_dim - self.time - 1):
        if set(combs_N[self.time]).issubset(set(combs_N[i + self.time])):
          self.mask[i + self.time] = 1
      self._state[self.time] = 1
    elif action == 0:
      self._state[self.time] = 0
    else:
      raise ValueError('`action` should be 0 or 1.')

    self._state[self.time + state_dim] = 0
    self.time += 1

    if self.time >= state_dim:
        self._episode_ended = True
    else:
      self._state[self.time + state_dim] = 1

    if self._episode_ended:
      self.reward = calcScore(self._state[:state_dim])
      return self._state, self.reward, True, {}
    else:
      return self._state, 0, False, {}

  def render(self):
    if (self.reward != 0):
      print(self.reward)

env = HyperGraphEnv()
model = PPO("MlpPolicy", env, verbose = 1)
model.learn(total_timesteps=9999999999999999999999999999)
obs = env.reset()

for i in range(1000):
  action, _states  = model.predict(obs,deterministic=True)
  obs, rewards, done, _  = env.step(action)
  env.render()
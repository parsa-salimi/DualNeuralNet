import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import utils
from tf_agents.networks.q_rnn_network import QRnnNetwork
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.specs import array_spec
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.networks import actor_distribution_network

from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.trajectories.trajectory import to_transition, from_transition
from tf_agents.utils.common import function

import itertools
from hypergraph import *
from params import *



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
  return len(primal)




class HyperGraphEnv(py_environment.PyEnvironment):


  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(2 * state_dim,), dtype=np.int32, minimum=0, name='observation')
    self._state = np.zeros(2 * state_dim)
    self._episode_ended = False
    self.time = 0
    self.mask = np.zeros(state_dim)

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = np.zeros(state_dim * 2)
    self._episode_ended = False
    self.time = 0
    self.mask = np.zeros(state_dim)
    return ts.restart(np.array(self._state, dtype=np.int32))

  def _step(self, action):

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()

    # Make sure episodes don't go on forever.
    if self.mask[self.time] == 1:
      self._state[self.time] == 0
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
      reward = calcScore(self._state[:state_dim])
      return ts.termination(np.array(self._state, dtype=np.int32), reward)
    else:
      return ts.transition(
          np.array(self._state, dtype=np.int32), reward=0.0, discount=1.0)


 


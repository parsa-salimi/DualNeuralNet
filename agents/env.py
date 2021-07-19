import numpy as np
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import utils
from tf_agents.specs import array_spec

n = 8
state_dim = 2**n

class HyperGraphEnv(py_environment.PyEnvironment):

  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(state_dim,), dtype=np.int32, minimum=0, name='observation')
    self._state = np.zeros(state_dim)
    self._episode_ended = False
    self.time = 0

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = np.zeros(state_dim)
    self._episode_ended = False
    self.time = 0
    return ts.restart(np.array(self._state, dtype=np.int32))

  def _step(self, action):

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()

    # Make sure episodes don't go on forever.
    
    if action == 1:
      self._state[self.time] = 1
    elif action == 0:
      self._state[self.time] = 0
    else:
      raise ValueError('`action` should be 0 or 1.')

    self.time += 1

    if self.time >= state_dim:
        self._episode_ended = True

    if self._episode_ended:
      reward = sum(self._state)
      return ts.termination(np.array(self._state, dtype=np.int32), reward)
    else:
      return ts.transition(
          np.array(self._state, dtype=np.int32), reward=0.0, discount=1.0)


env = HyperGraphEnv()
time_step = env.reset()
print(time_step)
reward = time_step.reward

for i in range(state_dim):
    #time_step = env.step(np.random.random_integers(0,1))
    time_step = env.step(1)
    print(time_step)
    reward += time_step.reward
    print(reward)

print(reward)


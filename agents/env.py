import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import utils
from tf_agents.networks.q_rnn_network import QRnnNetwork
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.specs import array_spec
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.trajectories.trajectory import to_transition
from tf_agents.utils.common import function
import logging
import itertools
from hypergraph import *
from params import *



state_dim = 2**N
list_N = range(1,N+1)
combs_N = []
for i in range(1,N):
	combs_N.extend(list(itertools.combinations(list_N,i)))

def calcScore(state):
	primal = []
	for (i,j) in zip(combs_N,range(state_dim -1)) :
		if (state[j] == 1):
			primal.append(i)

	primal = reduce(primal)
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

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = np.zeros(state_dim * 2)
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


env = HyperGraphEnv()
tf_env = TFPyEnvironment(env)

fc_layer_params=[128,128,64]
q_net = QRnnNetwork(tf_env.observation_spec(), tf_env.action_spec(), lstm_size=(16,))
q_net_2 = q_net = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    fc_layer_params=fc_layer_params)

#agent
train_step = tf.Variable(0)
#optimizer = tf.keras.optimizers.RMSprop(learning_rate=2.5e-4, rho=0.95, momentum=0.0, epsilon=0.00001, centered= True)
optimizer = tf.keras.optimizers.Adam(lr=0.0001)
decay_fn = tf.keras.optimizers.schedules.PolynomialDecay(
  initial_learning_rate = 1.0,
  decay_steps = 2500,
  end_learning_rate = 0.01
)
agent = DqnAgent(tf_env.time_step_spec(), 
                 tf_env.action_spec(),
                 q_network=q_net_2, 
                 optimizer = optimizer,
                 td_errors_loss_fn = tf.keras.losses.Huber(reduction="none"),
                 train_step_counter = train_step,
                 target_update_period = 100,
                 epsilon_greedy = lambda : decay_fn(train_step))
agent.initialize()

print(tf_env.batch_size)
print(agent.collect_data_spec)
print(tf_env.batch_size)

#replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
  data_spec = agent.collect_data_spec,
  batch_size = tf_env.batch_size,
  max_length = 1000000
)

replay_buffer_observer = replay_buffer.add_batch

#metrics
train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.MaxReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]


class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")



logging.getLogger().setLevel(logging.INFO)
log_metrics(train_metrics)

collect_driver = DynamicEpisodeDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_episodes=15)

initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
                                        tf_env.action_spec())
init_driver = DynamicEpisodeDriver(
    tf_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(20000)],
    num_episodes=1000) 
final_time_step, final_policy_state = init_driver.run()



tf.random.set_seed(9) # chosen to show an example of trajectory at the end of an episode



trajectories, buffer_info = next(iter(replay_buffer.as_dataset(
    sample_batch_size=2,
    num_steps= 2,
    single_deterministic_pass=False)))

time_steps, action_steps, next_time_steps = to_transition(trajectories)
time_steps.observation.shape

dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps= 2,
    num_parallel_calls=3).prefetch(3)

collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(
            iteration, train_loss.loss.numpy()), end="")
        if iteration % 100 == 0:
            log_metrics(train_metrics)



train_agent(n_iterations=50000)





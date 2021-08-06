
from env import *
from replayBuffer import *
from params import *


env = HyperGraphEnv()
tf_env = TFPyEnvironment(env)

#hypermaramters
fc_layer_params=[64,64,64,64,64,64]



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
  decay_steps = 25000,
  end_learning_rate = 0.03
)
tf_agent = DqnAgent(tf_env.time_step_spec(), 
                 tf_env.action_spec(),
                 q_network=q_net_2, 
                 optimizer = optimizer,
                 td_errors_loss_fn = tf.keras.losses.Huber(reduction="none"),
                 train_step_counter = train_step,
                 target_update_period = 100,
                 epsilon_greedy = lambda : decay_fn(train_step))
tf_agent.initialize()


#replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
  #data_spec = agent.collect_data_spec,
  data_spec = tf_agent.collect_data_spec,
  batch_size = tf_env.batch_size,
  max_length = replay_buffer_capacity
)
replay_buffer_observer = replay_buffer.add_batch

collect_driver = DynamicEpisodeDriver(
    tf_env,
    tf_agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_episodes=2)

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
tf_agent.train = function(tf_agent.train)

def train_agent(n_iterations):
    time_step = None
    policy_state = tf_agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = tf_agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(
            iteration, train_loss.loss.numpy()), end="")
        if iteration % 100 == 0:
            log_metrics(train_metrics)



train_agent(n_iterations=500000)

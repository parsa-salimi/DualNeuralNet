from env import *
from params import *
from replayBuffer import *
env = HyperGraphEnv()
#tf_env = TFPyEnvironment(env)
train_env = TFPyEnvironment(env)
eval_env = TFPyEnvironment(env)






actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

#optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
optimizer =tf.compat.v1.train.AdamOptimizer(learning_rate=0.00003)

train_step_counter = tf.compat.v2.Variable(0)

tf_agent = reinforce_agent.ReinforceAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network=actor_net,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step_counter)
tf_agent.initialize()


eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

#replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
  #data_spec = agent.collect_data_spec,
  data_spec = tf_agent.collect_data_spec,
  batch_size = train_env.batch_size,
  max_length = replay_buffer_capacity
)
replay_buffer_observer = replay_buffer.add_batch

collect_driver = DynamicEpisodeDriver(
    train_env,
    collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(200)] + train_metrics,
    num_episodes=3) 
final_time_step, final_policy_state = collect_driver.run()





#replace with Driver
def collect_episode(environment, policy, num_episodes):

  episode_counter = 0
  environment.reset()

  while episode_counter < num_episodes:
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)

    if traj.is_boundary():
      episode_counter += 1

#replace with metrics
def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  max_return = 0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    if episode_return > max_return:
      max_return = episode_return
    total_return += episode_return

  #avg_return = total_return / num_episodes
  return max_return
  #avg_return.numpy()[0]


# Please also see the metrics module for standard implementations of different
# metrics.



# (Optional) Optimize by wrapping some of the code in a graph using TF function.
tf_agent.train = function(tf_agent.train)

# Reset the train step
tf_agent.train_step_counter.assign(0)



for _ in range(num_iterations):

  # Collect a few episodes using collect_policy and save to the replay buffer.
  #collect_episode(
  #    train_env, tf_agent.collect_policy, collect_episodes_per_iteration)
  collect_driver.run(num_episodes= collect_episodes_per_iteration)

  # Evaluate the agent's policy once before training.
  #avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
  #returns = [avg_return]


  # Use data from the buffer and update the agent's network.
  experience = replay_buffer.gather_all()


  
  train_loss = tf_agent.train(experience)
  replay_buffer.clear()

  step = tf_agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))

  if step % eval_interval == 0:
    log_metrics(train_metrics)
    #avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    #print('step = {0}: Average Return = {1}'.format(step, avg_return))
    #returns.append(avg_return)
    
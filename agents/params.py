N = 4 #number of variables
state_dim= 2**N  -1

LEARNING_RATE = 0.00001 #Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.
n_sessions =1000 #number of new sessions per iteration
percentile = 93 #top 100-X percentile we are learning from
super_percentile = 94 #top 100-X percentile that survives to next iteration
INF = 1000000

#replay buffer params
replay_buffer_capacity = 2000000

#network params
fc_layer_params=[64,64,64,64,64,64]d

#REINFORCE params
num_iterations = 2500000000 # @param {type:"integer"}
collect_episodes_per_iteration = 3 # @param {type:"integer"}

learning_rate = 1e-2 # @param {type:"number"}
log_interval = 10 # @param {type:"integer"}
num_eval_episodes = 10 # @param {type:"integer"}
eval_interval = 50 # @param {type:"integer"}



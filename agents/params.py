N = 4 #number of variables
state_dim= 2**N  -1

LEARNING_RATE = 0.00001 #Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.
n_sessions =1000 #number of new sessions per iteration
percentile = 93 #top 100-X percentile we are learning from
super_percentile = 94 #top 100-X percentile that survives to next iteration
INF = 1000000


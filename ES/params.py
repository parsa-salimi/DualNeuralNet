import itertools

N = 5 #number of variables
MYN = 2**N - 1
len_game = MYN 
#N is the number of variables set in params.py
list_N = range(1,N+1)
combs_N = []
#list of all the nonempty subsets of {1,...,n}
for i in range(1,N+1):
	combs_N.extend(list(itertools.combinations(list_N,i)))

#The size of the alphabet. In this file we will assume this is 2. 
#There are a few things we need to change when the alphabet size is larger,
#such as one-hot encoding the input, and using categorical_crossentropy as a loss function.
#note that this will have to change if we decide to modify the code to consider all hyperedges.
n_actions = 2

LEARNING_RATE = 0.00001 #Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.
n_sessions =1000 #number of new sessions per iteration
percentile = 93 #top 100-X percentile we are learning from
super_percentile = 94 #top 100-X percentile that survives to next iteration
INF = 1000000

NEURONS_PER_LAYER = 128
LSTM_CELLS        = 128



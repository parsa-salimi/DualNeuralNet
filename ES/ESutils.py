"This code collects the functionality common to both the FNN and RNN implementations"

import params 
import random
import numpy as np
from reward import score
from hypergraph import reduce, dual


def convert(state):
	"""
	convert binary string representation of a hypergraph to the human readable list of tuples representation used
	in hypergraph.py. 
	:param state: the hypergraph as a list where each element in the list is a boolean
	:returns list of tuples representation of hypergraph
	"""
	hg = []
	for (i,j) in zip(params.combs_N,range(params.len_game)) :
		if (state[j] == 1):
			hg.append(i)
	return hg


def calcScore(state):
	"""
	Compute min(state) and use prespecified reward function (in rewards.py) to return the final reward.
	:param state the final state, which is a representation of the completed hypergraphs
	:returns score of final hypergraph
	"""

	hg = convert(state)
	hg = reduce(hg)
	return score(hg)


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
	"""
	Select states and actions from games that have rewards >= percentile
	:param states_batch: list of lists of states, states_batch[session_i][t]
	:param actions_batch: list of lists of actions, actions_batch[session_i][t]
	:param rewards_batch: list of rewards, rewards_batch[session_i]

	:returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions
	
	This function was mostly taken from https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
	"""
	counter = params.n_sessions * (100.0 - params.percentile) / 100.0
	reward_threshold = np.percentile(rewards_batch,params.percentile)

	elite_states = []
	elite_actions = []
	elite_rewards = []
	for i in range(len(states_batch)):
		if rewards_batch[i] >= reward_threshold-0.0000001:		
			if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
				for item in states_batch[i]:
					elite_states.append(item.tolist())
				for item in actions_batch[i]:
					elite_actions.append(item)			
			counter -= 1
	elite_states = np.array(elite_states, dtype = bool)	
	elite_actions = np.array(elite_actions, dtype = bool)	
	return elite_states, elite_actions
	
def select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90):
	"""
	Select all the sessions that will survive to the next generation
	Similar to select_elites function
	If this function is the bottleneck, it can easily be sped up using numba
	"""
	
	counter = params.n_sessions * (100.0 - params.percentile) / 100.0
	reward_threshold = np.percentile(rewards_batch,params.percentile)

	super_states = []
	super_actions = []
	super_rewards = []
	for i in range(len(states_batch)):
		if rewards_batch[i] >= reward_threshold-0.0000001:
			if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
				super_states.append(states_batch[i])
				super_actions.append(actions_batch[i])
				super_rewards.append(rewards_batch[i])
				counter -= 1
	super_states = np.array(super_states, dtype = bool)
	super_actions = np.array(super_actions, dtype = bool)
	super_rewards = np.array(super_rewards)
	return super_states, super_actions, super_rewards


#this class is designed to run an iteration of the training loop. The behaviour of the training iteration is slightly
#different when we use RNNs, so this class abstracts it away. Also RNNs generate sessions slightly differently, 
#so this is implemented seperately and passed to the class
class Iteration:
	def __init__(self,useRNN,model,generate_session):
		if (useRNN == True):
			self.observation_space = params.len_game
		else:
			self.observation_space = 2 * params.len_game
		self.super_states =  np.empty((0,params.len_game,self.observation_space), dtype = bool)
		self.super_actions = np.array([], dtype = bool)
		self.super_rewards = np.array([])
		self.myRand = random.randint(0,100)
		self.useRNN = useRNN
		self.model = model
		self.generate_session = generate_session
		self.iteration_number = 0


	def run(self):
		sessions = self.generate_session(self.model,params.n_sessions)
	
		states_batch = np.array(sessions[0], dtype = bool)
		actions_batch = np.array(sessions[1], dtype = bool)
		rewards_batch = np.array(sessions[2])
		states_batch = np.transpose(states_batch,axes=[0,2,1])
		states_batch = np.append(states_batch,self.super_states,axis=0)

		if self.iteration_number>0:
			actions_batch = np.append(actions_batch,np.array(self.super_actions),axis=0)	
		rewards_batch = np.append(rewards_batch,self.super_rewards)
		
		elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile=params.percentile) #pick the sessions to learn from
		super_sessions = select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=params.super_percentile) #pick the sessions to survive
	
		super_sessions = [(super_sessions[0][i], super_sessions[1][i], super_sessions[2][i]) for i in range(len(super_sessions[2]))]
		super_sessions.sort(key=lambda super_sessions: super_sessions[2],reverse=True)
	
		#RNNs expect the data to be split into time steps. So we reshape our array.
		if (self.useRNN == True):
			elite_states = elite_states.reshape((-1,params.len_game,1))
			elite_actions = elite_actions.reshape((-1,1))

		self.model.fit(elite_states, elite_actions) #learn from the elite sessions
	
		self.super_states = [super_sessions[i][0] for i in range(len(super_sessions))]
		self.super_actions = [super_sessions[i][1] for i in range(len(super_sessions))]
		self.super_rewards = [super_sessions[i][2] for i in range(len(super_sessions))]
	
		rewards_batch.sort()
		mean_all_reward = np.mean(rewards_batch[-100:])	
		mean_best_reward = np.mean(self.super_rewards)	

		#generate output
		print("\n" + str(self.iteration_number) +  ". Best individuals: " + str(np.flip(np.sort(self.super_rewards))))
		self.iteration_number += 1
	
		if (self.useRNN == True):
			networkIndicatorString= 'RNN'
		else:
			networkIndicatorString = 'FNN'
		if (self.iteration_number%5 == 1): #Write all important info to files every 5 iterations
			with open(  networkIndicatorString+'best_100_rewards_'+str(self.myRand)+'.txt', 'a') as f:
				f.write(str(mean_all_reward)+"\n")
			with open( networkIndicatorString+'best_elite_rewards_'+str(self.myRand)+'.txt', 'a') as f:
				f.write(str(mean_best_reward)+"\n")
			with open( networkIndicatorString+'best_episode_reward'+str(self.myRand)+'txt','a') as f:
				f.write(str(np.amax(self.super_rewards)) + "\n" )
		if (self.iteration_number%50 == 1):
			with open( networkIndicatorString+'best_species_'+str(self.myRand)+'.txt', 'w') as f:
				for item in self.super_actions:
					hg = reduce(convert(item))
					f.write(str(hg) + "\n" + str(dual(hg)) + "\n" + str(score(hg)))
					f.write("\n ============\n")


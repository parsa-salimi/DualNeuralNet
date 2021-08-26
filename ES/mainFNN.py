
"""
Authors : A Z Wagner, Parsa Salimi
This code modifies the code in A Z Wagner's "Constructions in combinatorics via neural networks" to work with hypergraphs.
It uses Feedforward neural networks.
This code works with tensorflow 2.4.1, python version 3.6, and numpy version 1.19.5.
 """

import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
import params
from ESutils import *




#size of each state/observation
#Leave this at 2*len_game for feedforward networks
# the first len_game letters encode our partial word (with zeros on
#the positions we haven't considered yet), and the next len_game bits one-hot encode which letter we are considering now.
#So e.g. [0,1,0,0,   0,0,1,0] means we have the partial word 01 and we are considering the third letter now.
observation_space = 2*params.len_game 	
#length of each episode				  
state_dim = (observation_space,)



#we use a feedforward network with relu activations
#dropout is not a good idea for RL tasks. see:
#https://ai.stackexchange.com/questions/8293/why-do-you-not-see-dropout-layers-on-reinforcement-learning-examples/8295
#We didn't add batch normalization layers in order to have a fair comparison with the DQN networks

model = keras.Sequential([
	keras.layers.Dense(params.NEURONS_PER_LAYER*2, activation="relu"),
	keras.layers.Dense(params.NEURONS_PER_LAYER, activation="relu"),
	keras.layers.Dense(params.NEURONS_PER_LAYER, activation="relu"),
	keras.layers.Dense(params.NEURONS_PER_LAYER, activation="relu"),
	keras.layers.Dense(params.NEURONS_PER_LAYER, activation="relu"),
	keras.layers.Dense(params.NEURONS_PER_LAYER, activation="relu"),
	keras.layers.Dense(params.NEURONS_PER_LAYER, activation="relu"),
	keras.layers.Dense(params.NEURONS_PER_LAYER, activation="relu"),
	keras.layers.Dense(1, activation="sigmoid"),
])


model.build((None, observation_space))
model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate = params.LEARNING_RATE)) 
print(model.summary())





####the ES algorithm is implemented below. 
def generate_session(agent, n_sessions):
	"""
	Play n_session games using agent neural network.
	Terminate when games finish 
	
	Code inspired by https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
	"""
	#this array stores the state of all the sessions, keeping the entire trajectory of states (one state per time step)
	#Since our alphabet is binary, we will save space by using the bool data type
	states =  np.zeros([n_sessions, observation_space, params.len_game], dtype=bool)
	actions = np.zeros([n_sessions, params.len_game], dtype = bool)
	state_next = np.zeros([n_sessions,observation_space], dtype = bool)
	prob = np.zeros(n_sessions)
	#initial one-hot encoding : we are considering the first edge.
	states[:,params.len_game,0] = 1
	step = 0
	total_score = np.zeros([n_sessions])
	while (True):
		step += 1		
		#use the neural network to predict the next action for all the sessions
		prob = agent.predict(states[:,:,step-1], batch_size = n_sessions) 
		for i in range(n_sessions):
			#sample from the output of the neural network
			if np.random.rand() < prob[i]:
				action = 1
			else:
				action = 0
			actions[i][step-1] = action
			state_next[i] = states[i,:,step-1]
			if (action > 0):
				state_next[i][step-1] = action		
			state_next[i][params.len_game + step-1] = 0
			#if this isn't the last step, one-hot encode the next edge
			if (step < params.len_game):
				state_next[i][params.len_game + step] = 1			
			terminal = step == params.len_game
			#calculate the score if the hypergraphs are complete.
			if terminal:
				total_score[i] = calcScore(state_next[i])
			if not terminal:
				states[i,:,step] = state_next[i]						
		if terminal:
			break
	return states,actions,total_score

FNNIterator = Iteration(useRNN = False, model = model, generate_session = generate_session)
for i in range(3000):
	FNNIterator.run()

import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
import params
from ESutils import *
			  
observation_space = params.len_game
state_dim = (observation_space,)

#Model structure: a sequential network with three hidden layers, sigmoid activation in the output.
#I usually used relu activation in the hidden layers but play around to see what activation function and what optimizer works best.
#It is important that the loss is binary cross-entropy if alphabet size is 2.
model = keras.Sequential([
	keras.layers.LSTM(params.LSTM_CELLS+38,input_shape = ((None,1)), return_sequences = True ),
	keras.layers.LSTM(params.LSTM_CELLS,input_shape = ((None,1)), return_sequences = True),
	keras.layers.LSTM(params.LSTM_CELLS,input_shape = ((None,1))),
	keras.layers.Dense(1,activation="sigmoid")])
model.compile(loss="binary_crossentropy", optimizer= keras.optimizers.Adam(learning_rate = params.LEARNING_RATE))
print(model.summary())

####RL algorithm is implemented below. 
def generate_session(agent, n_sessions):
	"""
	Play n_session games using agent neural network.
	Terminate when games finish 
	
	Code inspired by https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
	"""
	states =  np.zeros([n_sessions, observation_space, params.len_game], dtype=bool)
	actions = np.zeros([n_sessions, params.len_game], dtype = bool)
	state_next = np.zeros([n_sessions,observation_space], dtype = bool)
	prob = np.zeros(n_sessions)
	step = 0
	total_score = np.zeros([n_sessions])
	while (True):
		step += 1		
		data = states[:,0:step,step-1].reshape((n_sessions,step,1))
		prob = agent.predict(data, batch_size = n_sessions) 
		for i in range(n_sessions):
			if np.random.rand() < prob[i]:
				action = 1
			else:
				action = 0
			actions[i][step-1] = action
			state_next[i] = states[i,:,step-1]
			if (action > 0):
				state_next[i][step-1] = action					
			terminal = step == params.len_game
			if terminal:
				total_score[i] = calcScore(state_next[i])
			if not terminal:
				states[i,:,step] = state_next[i]			
		if terminal:
			break
	return states, actions, total_score		
	

RNNIterator = Iteration(useRNN = True, model = model, generate_session = generate_session)
for i in range(3000):
	RNNIterator.run()
			

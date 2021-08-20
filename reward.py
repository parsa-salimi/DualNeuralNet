from hypergraph import *
from params import *
import math

def FreqScore(primal):
	if (len(vars(primal)) < N ):
		return -INF
	d = dual(primal)
	myfreq =  maxTotalFreq(primal,d)

	sumlen = len(primal) + len(d)
	myScore = myfreq  
	return -myScore 

def FreqRatioScore(primal):
	if (len(vars(primal)) < N ):
		return -INF
	d = dual(primal)
	myfreq =  maxTotalFreq(primal,d)

	sumlen = len(primal) + len(d)
	myScore = myfreq*math.log(sumlen) 
	return -myScore 

def score(f):
	return FreqScore(f)
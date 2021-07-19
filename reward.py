from hypergraph import *
from params import *
import math

def FreqGuaranteeScore(primal):
	if (len(vars(primal)) < N ):
		return -INF
	d = dual(primal)
	myfreq =  maxTotalFreq(primal,d)

	sumlen = len(primal) + len(d)
	myScore = myfreq  
	return -myScore 

def score(f):
	return FreqGuaranteeScore(f)
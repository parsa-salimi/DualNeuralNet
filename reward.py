from hypergraph import *
from params import *

def FreqGuaranteeScore(primal):
	if (len(vars(primal)) < N ):
		return -INF
	dual = dual(primal)
	myfreq =  maxTotalFreq(primal,dual)

	sumlen = len(primal) + len(dual)
	myScore = myfreq * math.log(sumlen,2)  
	return -myScore 

def score(f):
	return FreqGuaranteeScore(f)
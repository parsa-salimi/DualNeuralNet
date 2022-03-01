from params import N

list_N = range(1,N+1)

def vars(f):
	flat_list = [item for sublist in f for item in sublist]
	return list(set(flat_list))

def remove_var(f,var):
	return [tuple(x for x in c if not(x == var)) for c in f]

def remove_clause(f,var):
	return [c for c in f if not(var in c)]

def reduce(f):
	def minimal(clause,f):
		if f==[]:
			return True
		for c in f:
			if ((not len(c) == len(clause)) and set(c).issubset(set(clause))):
				return False
		return True

	return list(set([c for c in f if minimal(c,f)])) 

def disjunction(f,g):
	return f + g

def mult(f,g):
	def mult_clause(c,g):
		return [tuple(set(x + c)) for x in g]
	accum = []
	for c in f:
		accum = disjunction(mult_clause(c,g) ,accum)
	return reduce(accum)

def nonsup(f,g):
	def nscond(clause):
		for x in g :
			if set(x).issubset(set(clause)):
				return False
		return True
	return list(filter(nscond, f))

def dual(f):
	if f==[]:
		return [()]
	if f==[()]:
		return []
	x 	= min(vars(f))
	f1 	= remove_var(f,x)
	f0 	= remove_clause(f,x)
	g0	= dual(reduce(disjunction(f0,f1)))
	g0o1= dual(f0)
	g1  = nonsup(g0o1,g0)
	return reduce(disjunction(g0, mult( [(x,)], g1))) 
		
	
def freq(func, var):
	count = 0
	for c in func:
		if (var in c):
			count = count + 1
	if len(func)==0:
		return 1
	return count/len(func)

def totalFreq(f,g,var):
	return max(freq(f,var), freq(g,var))
	
def maxFreq(func):
	return max([freq(func,var) for var in list_N])

def maxTotalFreq(f,g):
	return max([totalFreq(f,g,var) for var in list_N])
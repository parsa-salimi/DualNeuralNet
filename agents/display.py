from params import N, MYN
import numpy as np
import math
from reward import *
from hypergraph import *
import itertools

list_N = range(1,N+1)
combs_N = []
for i in range(1,N):
  combs_N.extend(list(itertools.combinations(list_N,i)))

print(MYN)

with open("best_species_txt_946.txt", "r") as states:
  states_str = states.read().replace("[","").replace('\n', "").replace("]","\n")
  #print(states_str)
  states_str = states_str.split("\n")
  #for s in states_str:
  #narray = np.fromstring(states_str, dtype=int, sep=" ")
  #for i in narray:
  #  print(i)


for s in states_str:
    state = [int(x) for x in s.split(' ')]
    primal = []
    for (i,j) in zip(combs_N,range(MYN - N + 1)):
      if (state[j + N - 1] == 1):
        if (state[len(i)] == 1):
          primal.append(i)

    primal = reduce(primal)
    print(primal)
    #d = dual(primal)
    #print(d)
    print(score(primal))
    print(len(primal))
    #print(totalMaxFreq(primal,d) * math.log((len(d) + len(primal)), 2))
    print("----------------------") 



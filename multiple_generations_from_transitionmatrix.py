#!/usr/bin/env python3


import os
import sys
import numpy as np

tranmx = np.array([[.6, .2, .1, 0], 
					[.2, .5, .1, .1], 
					[.1, .1, .5, .2], 
 					[0, .1, .2, .6]])

tranmx = tranmx.T

generation = 1 #first column of the matrix = first gen dispersal
mulgendis = tranmx[:,0]
mulgendisdil = tranmx[:,0]
genvector = tranmx[:,0] 

while generation < 14:
	#calculate where mussels from column 1 will be in the next generation
	genvector = genvector.dot(tranmx) 
	#add next generation to total multigenerational dispersal
	mulgendis = mulgendis + genvector 
	#dillute the amount of genes by number of generations(your grandkids have less of your DNA than your kids)
	mulgendisdil = mulgendisdil + genvector/2**generation 
	generation += 1

mulgendisP = mulgendis/sum(mulgendis)
mulgendisdilP = mulgendisdil/sum(mulgendisdil)

print("Dispersal total: ", mulgendis)
print("Dispersal proportions: ", mulgendisP)
print("Dilluted dispersal: ", mulgendisdil)
print("Dilluted dispersal proportions: ", mulgendisdilP) #this is the most meaningful output 

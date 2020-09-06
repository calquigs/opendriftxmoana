#!/usr/bin/env pyth


import os
import sys
import numpy as np

tranmx = np.array([[.6, .2, .1, 0], 
					[.2, .5, .1, .1], 
					[.1, .1, .5, .2], 
 					[0, .1, .2, .6]])

tranmx = tranmx.T

generation = 0
multigendispersal = np.empty(4)
multigendispersal_dilluted = np.empty(4)

while generation <= 14:
	genvector= (np.linalg.matrix_power(tranmx, generation)).dot(tranmx[:,0])
	multigendispersal = np.add(multigendispersal, genvector)

	genvector_dilluted = np.divide(genvector, 2**generation)
	multigendispersal_dilluted = np.add(multigendispersal_dilluted, genvector_dilluted)

	generation += 1

multigendispersalP = np.divide(multigendispersal, sum(multigendispersal))
multigendispersal_dillutedP = np.divide(multigendispersal_dilluted, sum(multigendispersal_dilluted))

print("Dispersal total: ", multigendispersal)
print("Dispersal proportions: ", multigendispersalP)
print("Dilluted dispersal: ", multigendispersal_dilluted)
print("Dilluted dispersal proportions: ", multigendispersal_dillutedP)

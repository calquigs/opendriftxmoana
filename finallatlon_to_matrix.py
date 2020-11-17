#!/usr/bin/env python

import os
import sys
import re
import numpy as np
import scipy as scipy
#from numpy import savetxt
from scipy import stats
from scipy import sparse
from scipy.sparse import csr_matrix
from matplotlib import colors
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

onesourcemat = np.empty((331, 300, 0))
lonsrange = [166, 184]
latsrange = [-47.5, -34]
bins = [331, 300]
nzrange = [lonsrange, latsrange]


f = open("C:/Users/quigleca/Desktop/opendrift_scripts/practicefile.txt", "r")

#get start/final lats/lons from file
starttofinish = ""
filelen = 0
for line in f:
	line = re.sub("\n", "\t", line)
	starttofinish += line
	filelen += 1

starttofinish = starttofinish.split("\t")
starttofinish = np.asarray(starttofinish)
starttofinish = np.reshape(starttofinish, (filelen, 4))
starttofinish = starttofinish.astype(float)

binnies = scipy.stats.binned_statistic_2d(
	starttofinish[:,2], starttofinish[:,3], starttofinish[:,2], 
	statistic = "count", 
	bins = bins, 
	range = nzrange
	)
	
#get percent settlers from total
binniesstat = (binnies.statistic)/filelen
print(binnies.statistic)
print("wtf")
onesourcemat = np.dstack((onesourcemat, binniesstat))

meanmat = np.mean(onesourcemat, 2)

#assembling bigmomma
from scipy.sparse import lil_matrix

bigmomma = csr_matrix((bins[0]*bins[1], bins[0]*bins[1]))


startbin = scipy.stats.binned_statistic_2d(
	startlon, startlat, startlon,
	statistic = "count",
	bins = bins,
	range = nzrange)

#exclude binedges
startbinnumber = 0
startbinnumber = (startbin.binnumber - 2*(startbinnumber // (bins[1]+2)) - bins[1]-1)
startbinnumber = startbinnumber[0]

bigmomma[startbinnumber, ] = meanmat.flatten('F')
sparse.save_npz("C:/Users/quigleca/Desktop/moana_outputs/bigmomma.npz", bigmomma)
bigmommaback = sparse.load_npz("C:/Users/quigleca/Desktop/moana_outputs/bigmomma.npz")
print(startbinnumber)
print(bigmommaback[startbinnumber,34501])

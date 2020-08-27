#!/usr/bin/env python

import os
import sys
import numpy as np
import scipy as scipy
#from numpy import savetxt
from scipy import stats
from matplotlib.image import NonUniformImage
from matplotlib import colors
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

onesourcemat = np.empty((300, 331, 0))
lonsrange = [166, 184]
latsrange = [-47.5, -34]
bins = [300, 331]
nzrange = [lonsrange, latsrange]

os.chdir("C:/Users/quigleca/Desktop/moana_outputs/processed")

for file in os.listdir("C:/Users/quigleca/Desktop/moana_outputs/processed"):
	f = open(file, "r")

	lons = [166, 184]
	lats = [-47.5, -34]

	#lons = [164.2506,183.7039]
	#lats = [-47.2785,-33.9572]
	startlon = []
	startlat = []

	#is the first line the source cell?
	firstlinestart = True

	#get lats/lons from file
	entrylons = []
	entrylats = []
	for line in f:
		if firstlinestart:
			elems = line.split()
			startlon = elems[0]
			startlat = elems[1]
			firstlinestart = False
		elems = line.split()
		entrylon = float(elems[0])
		entrylat = float(elems[1])
		#convert negative lons to positve
		if entrylon < 0:
			entrylon += 360
		entrylons.append(entrylon)
		entrylats.append(entrylat)
	#get number of particles
	particlecount = len(entrylons)

	#remove NaN
	entrylons = [x for x in entrylons if str(x) != "nan"]
	entrylats = [x for x in entrylats if str(x) != "nan"]

	#bins = 311
	#nzrange = [[164.2506,183.7039], [-47.2785,-33.9572]]

	binnies = scipy.stats.binned_statistic_2d(
		entrylons, entrylats, entrylons, 
		statistic = "count", 
		bins = bins, 
		range = nzrange
		)
	#binnies2 = binnies.statistic + 1
	binniesstat = (binnies.statistic)/particlecount
		
	onesourcemat = np.dstack((onesourcemat, binniesstat))

meanmat = np.mean(onesourcemat, 2)
print(binnies.binnumber)

#assembling bigmomma
from scipy.sparse import lil_matrix

bigmomma = lil_matrix((99300, 99300))




#inputlon = float(input("Destination longitude? "))
#if inputlon < 0:
#	inputlon += 360
#if inputlon >= 164.2506 and inputlon <= 183.7039:
#	pass
#else:
#	print("Longitude out of range")
#	sys.exit()
#
#inputlat = float(input("Destination latitude? "))
#if inputlat >= -47.2785 and inputlat <= -33.9572:
#	pass
#else:
#	print("Latitude out of range")
#	sys.exit()
#
#print("nice, thanks")
#
#
#inlonbin = np.digitize(inputlon, binnies.x_edge)
#print(inlonbin)
#inlatbin = np.digitize(inputlat, binnies.y_edge)
#print(inlatbin)
#
#print(meanmat[inlonbin, inlatbin])
#print(meanmat[inlonbin][inlatbin])
#testlons = [164.2506,183.7039]
#testlats = [-47.2785,-33.9572]

#testbins = scipy.stats.binned_statistic_2d(
#		testlons, testlats, testlons, 
#		statistic = "count", 
#		bins = bins, 
#		range = nzrange,
#		)
#print(testbins.binnumber)

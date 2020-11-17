#!/usr/bin/env python

import os
import sys
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

startlon = []
startlat = []
#only pull start lat lon once
getstartlatlon = True

os.chdir("C:/Users/quigleca/Desktop/moana_outputs/processed")

for file in os.listdir("C:/Users/quigleca/Desktop/moana_outputs/processed"):
	f = open(file, "r")

	#is the first line the source cell?
	firstlinestart = True

	#get lats/lons from file
	entrylons = []
	entrylats = []
	for line in f:
		if firstlinestart:
			elems = line.split()
			if getstartlatlon:
				startlon.append(float(elems[0]))
				startlat.append(float(elems[1]))
				getstartlatlon = False
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
	
	#get percent settlers from total
	binniesstat = (binnies.statistic)/particlecount
		
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

startbinnumber = 0
startbinnumber = (startbin.binnumber - 2*(startbinnumber // (bins[1]+2)) - bins[1]-1)
startbinnumber = startbinnumber[0]

bigmomma[startbinnumber, ] = meanmat.flatten('F')
print(len(meanmat.flatten("F")))
sparse.save_npz("C:/Users/quigleca/Desktop/moana_outputs/bigmomma.npz", bigmomma)
bigmommaback = sparse.load_npz("C:/Users/quigleca/Desktop/moana_outputs/bigmomma.npz")

print(bigmommaback[startbinnumber,34501])
#Flatten 2d array of destinations to 1d (maybe trim edges?)
#try binning with just one entry (startlat, startlon) to get binnumber to store in poroper column.

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

#!/usr/bin/env python3

import os
import sys
import numpy as np
import netCDF4 as nc
#import matplotlib.pyplot as plt
import math
#from mpl_toolkits.basemap import Basemap

#open + load  file
fn = "/Users/calvinquigley/Desktop/14mussels_1day_output.nc"
traj = nc.Dataset(fn)

#extract variables
lon = traj["lon"][:]
lat = traj["lat"][:]
status = traj["status"][:]
time = traj["time"][:]

#spawnstart = 7
#spawnend = 11
#starttime = getattr(traj, "time_coverage_start")
#starttime = datetime.strptime(starttime, "%Y-%m-%d %H:%M:%S")
#timestep = getattr(traj, "time_step_output")
#timestep = datetime.strptime(timestep, "%H:%M:%S").time()


#extract final timestep of each particle from status variable 
finaltimes = []
for part in status:
	finaltime = [i for i in range(len(part)) if part[i] > 0]
	finaltimes += finaltime

#extract start lon, lat for each particle (first nonmasked value)
startlon = []
startlat = []
for part in lon:
	loncomp = part.compressed()
	startlon.append(loncomp[0])

for part in lat:
	latcomp = part.compressed()
	startlat.append(latcomp[0])

#extract final lon,lat for each particle
finallon = []
finallat = []
count = 0
for final in finaltimes:
	#only include succesfully settled
	#if final == (whatever number):
	#clip spawning season
	#finaldt = starttime + final * timestep - age
	finallon.append(lon[count, final])
	finallat.append(lat[count, final])
	count += 1

#clip to startlon/lat of complete trajectories (ideally they're all complete)
startlon2 = []
startlat2 = []
count = 0

for i in finallon:
	if math.isnan(finallon[count]) == False:
		startlon2.append(startlon[count])
	if math.isnan(finallat[count]) == False:
		startlat2.append(startlat[count])
	count += 1


#clip to finallat/lon of complete trajetories
finallon = [x for x in finallon if math.isnan(x) == False]
finallat = [x for x in finallat if math.isnan(x) == False]

#convert lons/lats to points
import shapely
from shapely.geometry import Point
from shapely.geometry import shape
startpoints = []
count = 0
for i in startlon2:
	startpoints.append(Point(startlon2[count], startlat2[count]))
	count += 1

finalpoints = []
count = 0
for i in finallon:
	finalpoints.append(Point(finallon[count], finallat[count]))
	count += 1

#get bins from shapefile
import shapefile
shp = shapefile.Reader("Desktop/settlement_bins/settlement_bins.shp")
bins = shp.shapes()
records = shp.records()

#create empty connectivity matrix
conmat = np.empty((len(bins), len(bins)))

for i in range(len(startpoints)):
	for nbin in bins:
		if startpoints[i].within(shape(nbin)):
			print("Point ", startpoints[i]," starts in bin ",records[i])





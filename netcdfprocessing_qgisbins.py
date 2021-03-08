#!/usr/bin/env python3

import os
import sys
import numpy as np
import netCDF4 as nc
#import matplotlib.pyplot as plt
import math
#from mpl_toolkits.basemap import Basemap

#open + load  file
fn = "14mussels_1day_output.nc"
traj = nc.Dataset(fn)

#extract variables
lon = traj["lon"][:]
lat = traj["lat"][:]
status = traj["status"][:]
time = traj["time"][:]

#extract final timestep of each particle from status variable 
print(len(status))
#finaltimes = [ [i for i in range(len(part)) if part[i] > 0] for part in status ]
finaltimes = []
for part in status:
  print(f"status has {len(part)} parts {part.__class__}")
  finaltime = [i for i in range(len(part)) if part[i] > 0]
  print(part)
  finaltimes += finaltime
  break
print("done with finaltimes")
sys.exit(1)

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

#clip to startlon/lat of complete trajectories only (ideally they're all complete)
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
startbins = []
finalbins = []
conmat = np.empty((len(bins), len(bins)))

#check if start/end points fall in bins.
for i in range(len(startpoints)):
	for j in range(len(bins)):
		if startpoints[i].within(shape(nbin)):
			startbins.append((i,records[j][0]))
		
for i in range(len(finalpoints)):
	for j in range(len(bins)):
		if finalpoints[i].within(shape(nbin)):
			startbins.append((i,records[j][0]))
			
#check if point started AND ended in a bin

startbins2 = [i[1] for i in startbins if i[0] in finalbins[0,:]]
finalbins2 = [i[1] for i in finalbins if i[0] in startbins[0,:]]

startbins = startbins2
finalbins = finalbins2
		
					 





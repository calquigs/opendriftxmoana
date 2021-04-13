#!/usr/bin/env python
# 
import os
import sys
import numpy as np
import netCDF4 as nc

#open + load  file
fn = "Desktop/ncsmall.nc"
ds = nc.Dataset(fn)

#extract variables
lon = ds["longitude"][:]
lat = ds["latitude"][:]
status = ds["status"][:]

#extract final timestep of each particle from status variable 
finaltimes = []
for part in status:
	finaltime = [i for i in range(len(part)) if part[i] > 0]
	finaltimes += finaltime


#return final lon,lat for each particle
unodostres = 0
for final in finaltimes:
	print(str((lon[unodostres, final]))+", "+str((lat[unodostres, final])))
	unodostres += 1

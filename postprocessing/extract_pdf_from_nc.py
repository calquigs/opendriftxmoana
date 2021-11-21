#!/usr/bin/env python3

import os
import sys
import numpy as np
import netCDF4 as nc
import glob

site = sys.argv[1]

outFile = open(f'/nesi/nobackup/vuw03073/bigboy/pdfs/{site}_alltraj_flat.txt', 'w')
xs = np.array(())
ys = np.array(())
for file in sorted(glob.glob(f'/nesi/nobackup/vuw03073/bigboy/all_settlement/{site}*')):
	traj = nc.Dataset(file)
	print(file)
	lon = traj.variables['lon'][:]
	lat = traj.variables['lat'][:]
	x = lon[np.where(lon.mask==False)]
	y = lat[np.where(lat.mask==False)]
	xs = np.append(xs,x)
	ys = np.append(ys,y)
pts = np.array((xs,ys))
np.savetxt(outFile, pts)
outFile.close()

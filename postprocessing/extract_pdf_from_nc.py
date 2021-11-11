#!/usr/bin/env python3

import os
import sys
import numpy as np
import netCDF4 as nc

sites = ['OPO','MAU','WEST','FLE','TAS','LWR','CAP','CAM','KAI','GOB','TIM','HSB','BGB','FIO']

for site in sites:
	outFile = open(f'{site}_alltraj_flat.txt', 'w')
	xs = np.array(())
	ys = np.array(())
	for file in glob.glob(f'/nesi/nobackup/vuw03073/bigboy/all_settlement/{site}*'):
		traj = nc.Dataset(file)
		lon = traj.variables['lon'][:]
		lat = traj.variables['lat'][:]
		x = lon[np.where(lon.mask==False)]
		y = lat[np.where(lat.mask==False)]
		xs = np.append(xs,x)
		ys = np.append(ys,y)
	pts = np.array((xs,ys))
	np.savetxt(outFile, pts)
	outFile.close()
#!/usr/bin/env python3

import os
import sys
import glob
import numpy as np
import netCDF4 as nc
import xarray as xr
import geopy.distance


# sites = ['OPO','MAU','WEST','FLE','TAS','LWR','CAP','CAM','KAI','GOB','TIM','HSB','BGB','FIO']
# site = sites[sys.argv[1]]

# for site in sites:
# 	for file in sorted(glob.glob(f'/nesi/nobackup/vuw03073/bigboy/all_settlement/{site}*')):
# 		traj = nc.Dataset(file)
# 		ym = file[-9:-3]
# 		print(f'{site}_{ym}')
# 		lon = traj.variables['lon'][:]
# 		lat = traj.variables['lat'][:]
# 		dists = np.zeros((len(lon), 2))
# 		for part in range(len(lon)):
# 			part_lons = lon[part][np.where(lon[part].mask==False)]
# 			part_lats = lat[part][np.where(lat[part].mask==False)]
# 			along_track = 0
# 			for i in range(len(part_lons)-1):
# 				pos_t  = (part_lats[i], part_lons[i])
# 				pos_t2 = (part_lats[i+1], part_lons[i+1])
# 				along_track += geopy.distance.distance(pos_t, pos_t2).meters
# 			pos_0 = (part_lats[0], part_lons[0])
# 			pos_n = (part_lats[-1], part_lons[-1])
# 			start_finish = geopy.distance.distance(pos_0, pos_n).meters
# 			dists[part, 0] = along_track
# 			dists[part, 1] = start_finish
# 		outFile = open(f'bigboy_distances/{site}_{ym}.txt', 'w')
# 		np.savetxt(outFile, dists)
# 		outFile.close()

def calc_along_track(trajectory):
	import pdb; pdb.set_trace()
	lons = trajectory.lon.data
	lats = trajectory.lat.data
	lons = lons[(lons>-10000) & (lons < 10000)]
	lats = lats[(lats>-10000) & (lats < 10000)]
	along_track = 0
	for i in range(len(lons)-1):
		pos_t  = (lats[i], lons[i])
		pos_t2 = (lats[i+1], lons[i+1])
		along_track += geopy.distance.distance(pos_t, pos_t2).meters
	pos_0 = (lats[0], lons[0])
	pos_n = (lats[-1], lons[-1])
	start_finish = geopy.distance.distance(pos_0, pos_n).meters
	return along_track, start_finish


traj = xr.open_dataset(file_path)

xr.apply_ufunc(calc_along_track, 
	traj, 
	input_core_dims = [['time']], 
	output_core_dims=[['time']], 
	exclude_dims=set(['time']))
	#vectorize = True)

		




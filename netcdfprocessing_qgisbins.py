#!/usr/bin/env python3

import os
import sys
import numpy as np
import netCDF4 as nc
#import matplotlib.pyplot as plt
import math
#from mpl_toolkits.basemap import Basemap
from timeit import default_timer as timer

print('parse input file')
fn = "Desktop/14mussels_1day_output.nc"
traj = nc.Dataset(fn)

print('extract variables')
lon = traj["lon"][:]
lat = traj["lat"][:]
status = traj["status"][:]
time = traj["time"][:]

print('extract final timestep of each particle from status variable')
finaltimes = [np.where(part > 0)[-1] for part in status]
finaltimes = [t.item() if len(t) else -1 for t in finaltimes]

print('extract start lon, lat for each particle (first nonmasked value)')
startlon = [part.compressed()[0] for part in lon]
startlat = [part.compressed()[0] for part in lat]

print('extract final lon,lat for each particle')
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

print('clip to startlon/lat of complete trajectories only (ideally all complete)')
startlon2 = []
startlat2 = []
count = 0

for i in finallon:
    if math.isnan(finallon[count]) == False:
        startlon2.append(startlon[count])
    if math.isnan(finallat[count]) == False:
        startlat2.append(startlat[count])
    count += 1

print('clip to finallat/lon of complete trajetories')
finallon = [x for x in finallon if math.isnan(x) == False]
finallat = [x for x in finallat if math.isnan(x) == False]

print('converting lons/lats to points')
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

print('get bins from shapefile')
import shapefile
shp = shapefile.Reader("Desktop/settlement_bins/settlement_bins.shp")
bins = shp.shapes()
records = shp.records()

# The settlement cells are distributed only on the coastlines.
# To have O(1) lookups, we need to create a regular grid
# that just contains the indices of the "real" settlement bin
# that corresponds to the grid cell (if any).

class Grid:
    def __init__(self, bins):
        self.bins = bins
        min_lon = min([min([p[0] for p in b.points[:]]) for b in bins])
        max_lon = max([max([p[0] for p in b.points[:]]) for b in bins])
        min_lat = min([min([p[1] for p in b.points[:]]) for b in bins])
        max_lat = max([max([p[1] for p in b.points[:]]) for b in bins])
        print(f'lon range: ({min_lon}, {max_lon})')
        print(f'lat range: ({min_lat}, {max_lat})')
        self.lon_cell_size = 0.05
        self.lat_cell_size = 0.05
        nlon = round((max_lon - min_lon) / self.lon_cell_size)
        nlat = round((max_lat - min_lat) / self.lat_cell_size)
        print(f'grid size: ({nlon}, {nlat})')
        # save the origin of the grid
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.bin_idx = np.full((nlat, nlon), -1, dtype='int')
        # mark all the grid cells that have settlement bins
        for i in range(len(self.bins)):
            b = self.bins[i]
            lon_idx = self.lon_to_grid_col(b.points[0][0])
            lat_idx = self.lat_to_grid_row(b.points[0][1])
            self.bin_idx[lat_idx, lon_idx] = i
    def lon_to_grid_col(self, lon):
        return int((lon - self.min_lon) / self.lon_cell_size)
    def lat_to_grid_row(self, lat):
        return int((lat - self.min_lat) / self.lat_cell_size)
    def get_bin_idx(self, lon, lat):
        lon_idx = self.lon_to_grid_col(lon)
        lat_idx = self.lat_to_grid_row(lat)
        return self.bin_idx[lat_idx, lon_idx]

print('creating settlement-bin lookup grid...')
grid = Grid(bins)

print('checking if start/end points fall in bins')
startbins = []
for i in range(len(startpoints)):
    start_bin_idx = grid.get_bin_idx(startpoints[i].x, startpoints[i].y)
    if start_bin_idx < 0:
        continue  # didn't start within a bin
    startbins.append([i, records[start_bin_idx][0]])

finalbins = []
for i in range(len(finalpoints)):
    final_bin_idx = grid.get_bin_idx(finalpoints[i].x, finalpoints[i].y)
    if final_bin_idx < 0:
        continue  # didn't end within a bin
    finalbins.append([i, records[final_bin_idx][0]])
            
startfinalbins = []
for i in range(len(finalpoints)):
    start_bin_idx = grid.get_bin_idx(startpoints[i].x, startpoints[i].y)
    final_bin_idx = grid.get_bin_idx(finalpoints[i].x, finalpoints[i].y)
    startfinalbins.append([start_bin_idx, final_bin_idx])


print('check if point started AND ended in a bin')
# (MQ) it explodes here. Not sure what's supposed to happen...
startbins2 = [i[1] for i in startbins if i[0] in finalbins[0][:]]
finalbins2 = [i[1] for i in finalbins if i[0] in startbins[0][:]]

startbins = startbins2
finalbins = finalbins2
        
#create empty connectivity matrix
conmat = np.empty((len(bins), len(bins)+1))

#fill matrix!
for i in startfinalbins:
    if i[0] > 0:
        if i[1] == -1:
            conmat[i[0], -1] += 1
        else:
            conmat[i[0], i[1]] += 1

#convert to percent settlers
conmatpercent = np.empty((len(bins), len(bins)+1))
for i in range(len(conmat)-1):
    if sum(conmat[i]) > 0:
        conmatpercent[i] = conmat[i]/sum(conmat[i])

conmatpercent = conmatpercent[:,:-1]

#exclude empty rows
nonemptyr = 0
for i in conmatpercent:
    if sum(i) > 0:
        nonemptyr += 1

musmat = np.empty((nonemptyr, len(bins)))

count = 0
for i in range(len(conmatpercent)):
    if sum(conmatpercent[i] > 0):
        musmat[count] = conmatpercent[i]
        count += 1

#exclude empty columns
nonemptyc = 0
for i in range(len(musmat[0,:])):
    if sum(musmat[:,i]) > 0:
        nonemptyc +=1

musmattrim = np.empty((nonemptyr,nonemptyc))

count = 0 
for i in range(len(musmat[0,:])):
    if sum(musmat[:,i]) > 0:
        musmattrim[:, count] = musmat[:,i]
        count += 1



#create heatmap
import seaborn as sns
import matplotlib.pyplot as plt
ax = sns.heatmap(musmattrim, mask = (musmattrim == 0))


#create conmat based on regions
regionconmat = np.zeros((11, 12))

for i in startfinalbins:
    if i[0] > 0:
        if i[1] == -1:
            regionconmat[records[i[0]][5], -1] += 1
        else:
            regionconmat[records[i[0]][5]][records[i[1]][5]] += 1

#convert to percent settlers
regionconmatpercent = np.zeros((11, 12))
for i in range(10):
    if sum(regionconmat[i]) > 0:
        regionconmatpercent[i] = regionconmat[i]/sum(regionconmat[i])

regionconmatpercent = regionconmatpercent[:,:-1]

#assign region names to dataframe
import pandas as pd

labs = ['GLM9','GLM1','GLM2','GLM3_east','Chatham Islands', 'GLM3_west','Stewart Island', 'Auckland Islands','GLM7B','GLM7A','GLM8']
df = pd.DataFrame(data = regionconmatpercent, index = labs, columns = labs)


#create regional heatmap
ax = sns.heatmap(df, mask = (df == 0))





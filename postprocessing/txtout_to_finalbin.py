#!/usr/bin/env python3

import os
import sys
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import math
#from mpl_toolkits.basemap import Basemap
from timeit import default_timer as timer
import shapely
from shapely.geometry import Point, Polygon
from shapely.geometry import shape
import shapefile
import seaborn as sns
import pandas as pd
from mpl_toolkits.basemap import Basemap
import glob
from scipy.signal import convolve2d

def customout_to_startfinal_points(txt_in):
    inFile = open(txt_in, 'r')
    sfpoints = []
    for line in inFile:
        line = line.strip()
        elems = line.split(',')
        sfpoints.append([Point(float(elems[0]), float(elems[1])), Point(float(elems[2]), float(elems[3])), elems[4]])
    return sfpoints

bb = nc.Dataset('/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_200404.nc')

class Grid2:
    def __init__(self, lons, lats, bb):
        #create empty grid
        self.min_lon = lons[0]
        self.max_lon = lons[1]
        self.min_lat = lats[1]
        self.max_lat = lats[0]
        self.cell_size = .1
        self.nlon = round((self.max_lon - self.min_lon) / self.cell_size)
        self.nlat = round((self.max_lat - self.min_lat) / self.cell_size)
        print(f'grid size: ({self.nlon}, {self.nlat})')
        self.bin_idx = np.full((self.nlon, self.nlat), -1, dtype='int')
        #read mask from backbone
        mask = bb.variables['mask_rho'][:]
        lon_rho = bb.variables['lon_rho'][:]
        lat_rho = bb.variables['lat_rho'][:]
        #fill grid with mask
        for row in range(len(mask)):
            for column in range(len(mask[row,:])):
                if mask[row, column] == 0:
                    lon_idx = self.lon_to_grid_col(lon_rho[row, column])
                    lat_idx = self.lat_to_grid_row(lat_rho[row, column])
                    self.bin_idx[lon_idx, lat_idx] = 0
        #find adjacent bins
        kernel = [[0,1,0],[1,0,1],[0,1,0]]
        counts = convolve2d(self.bin_idx, kernel, mode='same', 
                    boundary='fill', fillvalue=-1)
        i = 1
        for row in range(len(self.bin_idx)):
            for column in range(len(self.bin_idx[row,:])):
                if counts[row, column] != -4 and self.bin_idx[row, column] == -1:
                    self.bin_idx[row, column] = i
                    i += 1
    def lon_to_grid_col(self, lon):
        if lon < 0:
            lon += 360
        if lon < self.min_lon:
            lon = self.min_lon
        if lon >= self.max_lon:
            lon = self.max_lon - .01
        return math.floor(round((lon - self.min_lon) / self.cell_size, 6))
    def lat_to_grid_row(self, lat):
        if lat < self.min_lat:
            lat = self.min_lat
        if lat >= self.max_lat:
            lat = self.max_lat - .01
        return math.floor(round((lat - self.min_lat) / self.cell_size, 6))
    def get_bin_idx(self, lon, lat):
        lon_idx = self.lon_to_grid_col(lon)
        lat_idx = self.lat_to_grid_row(lat)
        return self.bin_idx[lon_idx, lat_idx]
    def bin_corners(self, lon, lat):
        llon = self.min_lon + self.cell_size*self.lon_to_grid_col(lon)
        rlon = self.min_lon + self.cell_size*(self.lon_to_grid_col(lon)+1)
        blat = self.min_lat + self.cell_size*self.lat_to_grid_row(lat)
        tlat = self.min_lat + self.cell_size*(self.lat_to_grid_row(lat)+1)
        return [llon, rlon, blat, tlat]

grid = Grid2([164, 184], [-31, -52], bb)

for file in glob.glob('bigboy/*.txt'):
    outFile = open(f'bigboy/final_bins/{file[6:-4]}_finalbins.txt','w')
    sfpoints = customout_to_startfinal_points(file)
    for pt in sfpoints:
    	if grid.get_bin_idx(pt[1].x, pt[1].y)>0:
                outFile.write(str(grid.get_bin_idx(pt[1].x, pt[1].y))+'\n')
outFile.close()


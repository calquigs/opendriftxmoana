#!/usr/bin/env python3
print('why', flush=True)
import os
import sys
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import math
#from mpl_toolkits.basemap import Basemap
import shapely
from shapely.geometry import Point, Polygon
from shapely.geometry import shape
import shapefile
import pandas as pd
import glob

def nc_to_startfinal_points(nc_in,seed):
    '''
    Extract the start and final locations as shapely Points and end status 
    of each particle from an OpenDrift output .nc file.

    nc_in: the path to the .nc file
    '''
    #open file+extract variables
    traj = nc.Dataset(nc_in)
    lon = traj["lon"][:]
    lat = traj["lat"][:]
    status = traj["status"][:]
    
    finaltimes = [np.where(part > 0)[-1] for part in status]
    finaltimes = [t.item() if len(t) else -1 for t in finaltimes]
    #extract start lon, lat for each particle (first nonmasked value)
    seeds = np.loadtxt(seed)
    startlon = seeds[0]
    startlat = seeds[1]    
    #extract final lon,lat, and status for each particle
    finallon = []
    finallat = []
    finalstatus = []
    count = 0
    for final in finaltimes:
        finallon.append(lon[count, final])
        finallat.append(lat[count, final])
        finalstatus.append(status[count, final])
        count += 1
    #convert status flags to meanings
    statusmeanings = traj.variables['status'].flag_meanings
    statusmeanings = statusmeanings.split()
    finalstatus = [statusmeanings[finalstatus[i]] for i in range(len(finalstatus))]
    #convert start/final lon,lat to points
    startpoints = []
    for i in range(len(startlon)):
        startpoints.append(Point(startlon[i], startlat[i]))
    finalpoints = []
    for i in range(len(finallon)):
        finalpoints.append(Point(finallon[i], finallat[i]))
    sfpoints = []
    for i in range(len(startpoints)):
        sfpoints.append([startpoints[i], finalpoints[i], finalstatus[i]])
    return sfpoints

shape_filename = "/nesi/project/vuw03073/testScripts/all_reef_bins/all_reef_bins.shp"
shp = shapefile.Reader(shape_filename)
bins = shp.shapes()
records = shp.records()

class Grid:
    def __init__(self, bins, records):
        self.bins = bins
        min_lon = min([min([p[0] for p in b.points[:]]) for b in bins])
        max_lon = max([max([p[0] for p in b.points[:]]) for b in bins])
        min_lat = min([min([p[1] for p in b.points[:]]) for b in bins])
        max_lat = max([max([p[1] for p in b.points[:]]) for b in bins])
        print(f'lon range: ({min_lon}, {max_lon})')
        print(f'lat range: ({min_lat}, {max_lat})')
        self.lon_cell_size = 0.1
        self.lat_cell_size = 0.1
        self.nlon = round((max_lon - min_lon) / self.lon_cell_size)
        self.nlat = round((max_lat - min_lat) / self.lat_cell_size)
        print(f'grid size: ({self.nlon}, {self.nlat})')
        # save the origin of the grid
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.max_lat = max_lat
        self.bin_idx = np.full((self.nlon, self.nlat), -1, dtype='int')
        # mark all the grid cells that have settlement bins
        for i in range(len(self.bins)):
            b = self.bins[i]
            lon_idx = self.lon_to_grid_col(min([p[0] for p in b.points[:]]))
            lat_idx = self.lat_to_grid_row(min([p[1] for p in b.points[:]]))
            self.bin_idx[lon_idx, lat_idx] = records[i][4]
        # assign region to each grid cell
        self.bin_regions = np.full((self.nlon, self.nlat), -1, dtype='int')
        for i in range(len(self.bins)):
            b = self.bins[i]
            lon_idx = self.lon_to_grid_col(min([p[0] for p in b.points[:]]))
            lat_idx = self.lat_to_grid_row(min([p[1] for p in b.points[:]]))
            self.bin_regions[lon_idx, lat_idx] = records[i][3]
    def lon_to_grid_col(self, lon):
        if lon < 0:
            lon += 360
        if lon < self.min_lon:
            lon = self.min_lon
        if lon > self.max_lon:
            lon = self.max_lon - .01
        return math.floor(round((lon - self.min_lon) / self.lon_cell_size, 6))
    def lat_to_grid_row(self, lat):
        if lat < self.min_lat:
            lat = self.min_lat
        if lat > self.max_lat:
            lat = self.max_lat - .01
        return math.floor(round((lat - self.min_lat) / self.lat_cell_size, 6))
    def get_bin_idx(self, lon, lat):
        lon_idx = self.lon_to_grid_col(lon)
        lat_idx = self.lat_to_grid_row(lat)
        return self.bin_idx[lon_idx, lat_idx]
    def get_bin_region(self, lon, lat):
        lon_idx = self.lon_to_grid_col(lon)
        lat_idx = self.lat_to_grid_row(lat)
        return self.bin_regions[lon_idx, lat_idx]
    def plot_with_query_point(self, query_point):
        for bidx in range(len(self.bins)):
            b = self.bins[bidx]
            x = [p[0] for p in b.points]
            y = [p[1] for p in b.points]
            # # find ourselves in the grid
            # for col in range(self.nlon):
            #     for row in range(self.nlat):
            #         if self.bin_idx[row, col] == bidx:
            #             x.append(self.min_lon + (col + 0.5) * self.lon_cell_size)
            #             y.append(self.min_lat + (row + 0.5) * self.lat_cell_size)
            plt.plot(x, y)
            # break
        plt.scatter(query_point.x, query_point.y, s=80, marker="*")
        plt.show()

grid = Grid(bins, records)


def points_to_binmatrix(matrix, sfpoints):
    '''
    Build connectivity matrix from start+final points of trajectories using bin IDs.

    matrix: a 2d np.array of size (n, n+1), where n is equal to len(bins).
    sfpoints: a list of lists x where each element of x is a list that contains
    the start and final locations as shapely Points and end status for each particle.
    e.g. x = [[Point,Point, status],[Point, Point, status]]
    '''
    for x in sfpoints:
        sfbins = []
        for i in range(len(x)):
            start_bin_idx = grid.get_bin_idx(x[i][0].x, x[i][0].y) +1
            final_bin_idx = grid.get_bin_idx(x[i][1].x, x[i][1].y) +1
            print(x[i][0].x,x[i][0].y,start_bin_idx)
            sfbins.append([start_bin_idx, final_bin_idx])
        for i in sfbins:
            if i[0] != 0:
                if i[1] == -1:
                    matrix[i[0]-1, -1] += 1
                else:
                    matrix[i[0]-1, i[1]-1] += 1
    return matrix





sfpoints = [] 

region = sys.argv[1]

print(region, flush=True)

ncs = sorted(glob.glob(f'/nesi/nobackup/vuw03073/bigmomma/{region}*'))
seeds = sorted(glob.glob(f'/nesi/nobackup/vuw03073/bigmomma_seeds/{region}*'))

for i in range(len(ncs)):
	path_nc = ncs[i]
	path_seed = seeds[i]
	sfpoints.append(nc_to_startfinal_points(path_nc,path_seed))
print('building matrix', flush=True)
matrix = np.zeros((len(grid.bins), len(grid.bins)+1))
bigmomma = points_to_binmatrix(matrix, sfpoints)
print('matrix built', flush=True)

print('writing file', flush=True)
outFile = open(f'bigmomma_{region}.txt', 'w')
np.savetxt(outFile, bigmomma)
outFile.close()
print('file written', flush=True)

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

######################
#extract data from .nc
######################
def nc_to_startfinal_points(nc_in):
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
    startlon = [part.compressed()[0] for part in lon]
    startlat = [part.compressed()[0] for part in lat]
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

def customout_to_startfinal_points(txt_in):
    inFile = open(txt_in, 'r')
    sfpoints = []
    for line in inFile:
        line = line.strip()
        elems = line.split(',')
        sfpoints.append([Point(float(elems[0]), float(elems[1])), Point(float(elems[2]), float(elems[3])), int(elems[4])])
    return sfpoints


########################################
#read in settlement bins and create grid
########################################
shape_filename = "rho_settlement_bins/rho_settlement_bins.shp"
shp = shapefile.Reader(shape_filename)
bins = shp.shapes()
records = shp.records()

pts = bins[0].points
pts = [list(elem) for elem in pts]
poly = Polygon(pts)
point = Point(lon,lat)
poly.contain(point)
class Grid:
    def __init__(self, bins, records):
        self.bins = bins
        min_lon = min([min([p[0] for p in b.points[:]]) for b in bins])
        max_lon = max([max([p[0] for p in b.points[:]]) for b in bins])
        min_lat = min([min([p[1] for p in b.points[:]]) for b in bins])
        max_lat = max([max([p[1] for p in b.points[:]]) for b in bins])
        print(f'lon range: ({min_lon}, {max_lon})')
        print(f'lat range: ({min_lat}, {max_lat})')
        self.lon_cell_size = 0.05
        self.lat_cell_size = 0.05
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
            self.bin_idx[lon_idx, lat_idx] = i
        # assign region to each grid cell
        self.bin_regions = np.full((self.nlon, self.nlat), -1, dtype='int')
        for i in range(len(self.bins)):
            b = self.bins[i]
            lon_idx = self.lon_to_grid_col(min([p[0] for p in b.points[:]]))
            lat_idx = self.lat_to_grid_row(min([p[1] for p in b.points[:]]))
            self.bin_regions[lon_idx, lat_idx] = records[i][6]-1
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

#checking if start/end points fall in bins
missed_bin_count = 0
startbins = []
for i in range(len(startpoints)):
    start_bin_idx = grid.get_bin_idx(startpoints[i].x, startpoints[i].y)
    #print(f'looking up ({startpoints[i].x}, {startpoints[i].y})')
    if start_bin_idx < 0:
        # continue  # didn't start within a bin
        missed_bin_count += 1
        if missed_bin_count < 1000:
            continue  # ignore the first few, so we can look at them one-at-a-time
        print(f'point {i} did not start in a bin')
        min_dist = 1e9
        min_bin = None
        for b in bins:
            lats = [p[0] for p in b.points[:]]
            lons = [p[1] for p in b.points[:]]
            dlats = [(startpoints[i].x - lat) for lat in lats]
            dlons = [(startpoints[i].y - lon) for lon in lons]
            dists = [math.sqrt(dlats[i]*dlats[i] + dlons[i]*dlons[i]) for i in range(len(dlats))]
            if min(dists) < min_dist:
                min_dist = min(dists)
                min_bin = b
        print(f'query point: {startpoints[i]}')
        print(f'closest bin: {min_bin.points}')
        grid.plot_with_query_point(startpoints[i])
        sys.exit(1)
    startbins.append([i, records[start_bin_idx][0]])

################
#create matrices
################
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
            start_bin_idx = grid.get_bin_idx(x[i][0].x, x[i][0].y)
            final_bin_idx = grid.get_bin_idx(x[i][1].x, x[i][1].y)
            sfbins.append([start_bin_idx, final_bin_idx])
        for i in sfbins:
            if i[0] > 0:
                if i[1] == -1:
                    matrix[i[0], -1] += 1
                else:
                    matrix[i[0], i[1]] += 1
    for i in range(len(matrix)-1):
        if sum(matrix[i]) > 0:
            matrix[i] = matrix[i]/sum(matrix[i])
    matrix = matrix[:,:-1]
    return matrix


def points_to_regionmatrix(matrix, sfpoints):
    '''
    Build connectivity matrix from start+final points of trajectories using settlement regions.

    matrix: a 2d np.array of size (n, n+1), where n is equal to len(regions).
    sfpoints: a list of lists x where each element of x is a list that contains
    the start and final locations as shapely Points and end status for each particle.
    e.g. x = [[Point,Point, status],[Point, Point, status]]
    '''
    for x in sfpoints:
        sfregions = []
        for i in range(len(x)):
            start_bin_region = grid.get_bin_region(x[i][0].x, x[i][0].y)
            final_bin_region = grid.get_bin_region(x[i][1].x, x[i][1].y)
            sfregions.append([start_bin_region, final_bin_region])
        for i in sfregions:
            if i[0] > 0:
                if i[1] == -1:
                    matrix[i[0], -1] += 1
                else:
                    matrix[i[0], i[1]] += 1
    for i in range(len(matrix)-1):
        if sum(matrix[i]) > 0:
            matrix[i] = matrix[i]/sum(matrix[i])
    matrix = matrix[:,:-1]
    return matrix


#assign region lablels
quotalabs = ['GLM9','GLM1','GLM2','GLM3_east','Chatham Islands', 'GLM3_west','Stewart Island', 'Auckland Islands','GLM7B','GLM7A','GLM8']
regionlabs = ['waikato', '90milebeach_northland', 'hauraki_northland', 'bay_of_plenty', 'hawkes_bay', 'wellington_wairarapa', 'canterbury', 'chatham islands', 'otago', 'southland', 'stewart island', 'auckland islands', 'fiordland', 'west_coast', 'nelson_marlborough', 'wanganui', 'taranaki']
df = pd.DataFrame(data = y, index = regionlabs, columns = regionlabs)

#plot matrix
ax = sns.heatmap(df, mask = (df == 0))

############
#plot points
############
def plot_sfpoints(sfpoints):
    '''
    Plot start and final position of each particle.

    sfpoints: a list of lists x where each element of x is a list that contains
    the start and final locations as shapely Points and end status for each particle.
    e.g. x = [[Point,Point, status],[Point, Point, status]]
    '''
    for x in sfpoints:    
        for i in x:
            sx = i[0].x
            sy = i[0].y
            fx = i[1].x
            fy = i[1].y
            plt.plot(sx, sy, 'go')
            if i[2] == 0:
                plt.plot(fx, fy, 'bo')
            if i[2] == 1:
                plt.plot(fx, fy, 'yo')
            if i[2] == 'settled_on_coast':
                plt.plot(fx, fy, 'ro')
            if i[2] == 'settled_on_bottom':
                plt.plot(fx, fy, 'mo')
    m = Basemap(resolution= 'h', llcrnrlon = 162, llcrnrlat = -51, urcrnrlon = 186, urcrnrlat = -32)
    m.drawcoastlines()
    plt.show()



#############
#just for fun
#############
#exclude empty rows
nonemptyr = 0
for i in matrix:
    if sum(i) > 0:
        nonemptyr += 1

matrixr = np.empty((nonemptyr, len(bins)))

count = 0
for i in range(len(matrix)):
    if sum(matrix[i] > 0):
        matrixr[count] = matrix[i]
        count += 1

#exclude empty columns
nonemptyc = 0
for i in range(len(matrixr[0,:])):
    if sum(matrixr[:,i]) > 0:
        nonemptyc +=1

matrixtrim = np.empty((nonemptyr,nonemptyc))

count = 0 
for i in range(len(matrixr[0,:])):
    if sum(matrixr[:,i]) > 0:
        matrixtrim[:, count] = matrixr[:,i]
        count += 1


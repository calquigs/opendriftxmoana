#!/usr/bin/env python

import os
import sys
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
import random
from scipy import stats

txt_in = sys.argv[1]
print(txt_in)

######################
#extract data from .nc
######################
def nc_to_startfinal_points(nc_in):
    '''
    Return the start and final locations as shapely Points and end status 
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
shape_filename = "/nesi/project/vuw03073/testScripts/rho_settlement_bins/rho_settlement_bins.shp"
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
        #self.bin_regions = np.full((self.nlon, self.nlat), -1, dtype='int')
        #for i in range(len(self.bins)):
        #    b = self.bins[i]
        #    lon_idx = self.lon_to_grid_col(min([p[0] for p in b.points[:]]))
        #    lat_idx = self.lat_to_grid_row(min([p[1] for p in b.points[:]]))
        #    self.bin_regions[lon_idx, lat_idx] = records[i][6]-1
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
    #def get_bin_region(self, lon, lat):
    #    lon_idx = self.lon_to_grid_col(lon)
    #    lat_idx = self.lat_to_grid_row(lat)
    #    return self.bin_regions[lon_idx, lat_idx]
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

###################
#build refernce PDD
###################

def pdd(sfpoints):
    '''
    Return a 2d np.array representing a particle density distribution for a 
    list of trajectores that start in the same bin.
    sfpoints: a list of lists x where each element of x is a list that contains
    the start and final locations as shapely Points and end status for each particle.
    e.g. x = [[Point,Point, status],[Point, Point, status]].
    '''
    pdd = np.zeros((grid.nlon, grid.nlat))
    for i in sfpoints:
        pdd[grid.lon_to_grid_col(i[1].x),grid.lat_to_grid_row(i[1].y)] += 1
    #gaussian filtering??
    return pdd

full_run_path = ""

#full_run_points = nc_to_startfinal_points(full_run_path)
#ref_pdd = pdd(full_run_points)

def plot_pdd(sfpoints):
    x = [sfpoints[i][1].x for i in range(len(sfpoints))]
    y = [sfpoints[i][1].y for i in range(len(sfpoints))]
    lon_edges = np.linspace(165, 184, 381)
    lat_edges = np.linspace(-52, -32, 401)
    H, _, _ = np.histogram2d(y, x, [lat_edges, lon_edges], density = True)
    H[H==0]=None
    lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)
    m = Basemap(resolution= 'h', llcrnrlon = 165, llcrnrlat = -52, urcrnrlon = 184, urcrnrlat = -32)
    #m = Basemap(resolution= 'h', llcrnrlon = 170, llcrnrlat = -36, urcrnrlon = 175, urcrnrlat = -33)
    m.drawcoastlines()
    m.fillcontinents(color='white')
    xs, ys = m(lon_bins_2d, lat_bins_2d)
    plt.pcolormesh(xs, ys, H, cmap = 'inferno')
    #plt.show()


def compare_pdd_plots(sfpoints, ns):
    fig, axes = plt.subplots(2,2)
    axes = axes.flatten()
    count = 0
    for n in ns:
        x = [sfpoints[i][1].x for i in range(len(random.sample(sfpoints,n)))]
        y = [sfpoints[i][1].y for i in range(len(random.sample(sfpoints,n)))]
        lon_edges = np.linspace(165, 184, 381)
        lat_edges = np.linspace(-52, -32, 401)
        H, _, _ = np.histogram2d(y, x, [lat_edges, lon_edges], density = True)
        H[H==0]=None
        lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)
        axes[count].set_title(f"{n} particles")
        m = Basemap(resolution= 'h', llcrnrlon = 169, llcrnrlat = -46, urcrnrlon = 179, urcrnrlat = -41, ax = axes[count])
        #m = Basemap(resolution= 'h', llcrnrlon = 165, llcrnrlat = -52, urcrnrlon = 184, urcrnrlat = -32, ax = axes[count])
        m.drawcoastlines()
        m.fillcontinents(color='white')
        xs, ys = m(lon_bins_2d, lat_bins_2d)
        axes[count].pcolormesh(xs, ys, H, cmap = 'jet')
        axes[count].add_patch(Rectangle((170.75, -46), .2,.2, edgecolor = 'green', fill=False))
        count += 1
    x = [sfpoints[i][1].x for i in range(len(sfpoints))]
    y = [sfpoints[i][1].y for i in range(len(sfpoints))]
    lon_edges = np.linspace(165, 184, 381)
    lat_edges = np.linspace(-52, -32, 401)
    H, _, _ = np.histogram2d(y, x, [lat_edges, lon_edges], density = True)
    H[H==0]=None
    lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)
    axes[count].set_title(f"{len(sfpoints)} particles")
    #m = Basemap(resolution= 'h', llcrnrlon = 170, llcrnrlat = -36, urcrnrlon = 175, urcrnrlat = -33)
    m = Basemap(resolution= 'h', llcrnrlon = 169, llcrnrlat = -46, urcrnrlon = 179, urcrnrlat = -41, ax = axes[count])
    m.drawcoastlines()
    m.fillcontinents(color='white')
    xs, ys = m(lon_bins_2d, lat_bins_2d)
    axes[count].pcolormesh(xs, ys, H, cmap = 'jet')
    axes[count].add_patch(Rectangle((170.75, -46), .2,.2, edgecolor = 'green', fill=False))
    fig.suptitle('Dunedin, June 2016')

#######################
#bootstrap from ref_pdd
#######################

def sub_pdd(sfpoints, n):
    '''
    Return a 2d np.array representing a particle density distribution for a 
    random subsample of a list of trajectories that started in the same bin.
    sfpoints: a list of lists x where each element of x is a list that contains
    the start and final locations as shapely Points and end status for each particle.
    e.g. x = [[Point,Point, status],[Point, Point, status]].
    n: the number of particles to subsample
    '''
    sub_points = random.sample(sfpoints, n)
    sub_pdd = pdd(sub_points)
    return sub_pdd

def fuv(ref_pdd, sub_pdd0):
    '''
    Return the fraction of unexplained variance (fuv = 1 - r**2, where r is 
    the linear correlation coefficiant) between two particle density distributions.
    ref_pdd and sub_pdd: 2d np.arrays representing particle density distributions
    '''
    ref_pdd = ref_pdd.flatten()
    sub_pdd0 = sub_pdd0.flatten()
    r = stats.pearsonr(ref_pdd, sub_pdd0)
    fuv = 1 - r[0]**2
    return fuv

def boot_fuv(sfpoints, n, boot):
    '''
    Return a list of fuv values between a refernce pdd and a given 
    number of bootstrapped subsamples.
    sfpoints: a list of lists x where each element of x is a list that contains
    the start and final locations as shapely Points and end status for each particle.
    e.g. x = [[Point,Point, status],[Point, Point, status]].
    n: the number of particles to subsample
    boot: the number of bootstraps to perform
    '''
    ref_pdd = pdd(sfpoints)
    boot_fuv = []
    while boot > 0:
    	sub_pdd0 = sub_pdd(sfpoints, n)
    	fuv0 = fuv(ref_pdd, sub_pdd0)
    	boot_fuv.append(fuv0)
    	boot -= 1
    return boot_fuv


##################
#create fuv curves
##################

def fuv_data(sfpoints, ns, boot):
    '''
    Return a 2d np.array of the mean fuv for different numbers of 
    subsampled particles bootstrapped a given number of times.
    sfpoints: a list of lists x where each element of x is a list that contains
    the start and final locations as shapely Points and end status for each particle.
    e.g. x = [[Point,Point, status],[Point, Point, status]].
    ns: a list containing the different number of particles to be tested
    boot: the number of bootstraps to perform for each n
    '''
    fuv_data = np.zeros((2, len(ns)))
    for i in range(len(ns)):
    	fuv = boot_fuv(sfpoints, ns[i], boot)
    	fuv_data[0][i] = ns[i]
    	fuv_data[1][i] = np.max(fuv)
    return fuv_data

def fuv_curve(fuv_data0, label):
    '''
    Plot the output of fuv_data.
    '''
    x = fuv_data0[0]
    f = fuv_data0[1]
    g = [.05]*len(x)
    g = np.array(g)
    #calculate point after where fuv drops below .05
    idx = np.argwhere(np.diff(np.sign(f-g))).flatten()
    idx += 1
    plt.plot(x, f, '-', label = label)
    plt.plot(x, g, '--k')
    #plt.plot(x[idx], g[idx], 'ro')
    plt.legend()
    #plt.annotate(f"n = {x[idx][0]}", (x[idx], g[idx]))
    plt.xlabel('Number of particles released per month')
    plt.ylabel('Max fraction of unexplained variance (1-r**2)')
    #plt.show()


ns = [100,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000]

sfpoints = customout_to_startfinal_points(txt_in)
fuv_data0 = fuv_data(sfpoints, ns, 100)

outFile = open(f'dunedin_fuv_{txt_in[-9:-3]}.txt', 'w')
outFile.write(fuv_data0)
outFile.close()
outFile = open(f'reinga_fuv/reinga_fuv_{txt_in[-10:-4]}.txt', 'w')
for row in fuv_data0:
    np.savetxt(outFile, row)

outFile.close()


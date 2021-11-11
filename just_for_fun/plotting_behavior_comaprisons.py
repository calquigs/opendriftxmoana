#!/usr/bin/env python

import os
import sys
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math

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
from scipy.signal import convolve2d
from matplotlib import colors


LWR_passive_201601 = nc_to_startfinal_points('behavior_test/passive/LWR_201601.nc')
LWR_passive_201602 = nc_to_startfinal_points('behavior_test/passive/LWR_201602.nc')
LWR_passive_201603 = nc_to_startfinal_points('behavior_test/passive/LWR_201603.nc')
LWR_passive_201604 = nc_to_startfinal_points('behavior_test/passive/LWR_201604.nc')
LWR_passive_201605 = nc_to_startfinal_points('behavior_test/passive/LWR_201605.nc')
LWR_passive_201606 = nc_to_startfinal_points('behavior_test/passive/LWR_201606.nc')
LWR_passive_201607 = nc_to_startfinal_points('behavior_test/passive/LWR_201607.nc')
LWR_passive_201608 = nc_to_startfinal_points('behavior_test/passive/LWR_201608.nc')
LWR_passive_201609 = nc_to_startfinal_points('behavior_test/passive/LWR_201609.nc')
LWR_passive_201610 = nc_to_startfinal_points('behavior_test/passive/LWR_201610.nc')
LWR_passive_201611 = nc_to_startfinal_points('behavior_test/passive/LWR_201611.nc')
LWR_passive_201612 = nc_to_startfinal_points('behavior_test/passive/LWR_201612.nc')

lwr_passive = LWR_passive_201601 + LWR_passive_201602 + LWR_passive_201603 + LWR_passive_201604 + LWR_passive_201605 + LWR_passive_201606 + LWR_passive_201607 + LWR_passive_201608 + LWR_passive_201609 + LWR_passive_201610 + LWR_passive_201611 + LWR_passive_201612

LWR_depth_reg_201601 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201601.nc')
LWR_depth_reg_201602 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201602.nc')
LWR_depth_reg_201603 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201603.nc')
LWR_depth_reg_201604 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201604.nc')
LWR_depth_reg_201605 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201605.nc')
LWR_depth_reg_201606 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201606.nc')
LWR_depth_reg_201607 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201607.nc')
LWR_depth_reg_201608 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201608.nc')
LWR_depth_reg_201609 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201609.nc')
LWR_depth_reg_201610 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201610.nc')
LWR_depth_reg_201611 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201611.nc')
LWR_depth_reg_201612 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201612.nc')

lwr_depth_reg = LWR_depth_reg_201601 + LWR_depth_reg_201602 + LWR_depth_reg_201603 + LWR_depth_reg_201604 + LWR_depth_reg_201605 + LWR_depth_reg_201606 + LWR_depth_reg_201607 + LWR_depth_reg_201608 + LWR_depth_reg_201609 + LWR_depth_reg_201610 + LWR_depth_reg_201611 + LWR_depth_reg_201612

LWR_depth_reg_201601 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201601.nc')
LWR_depth_reg_201602 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201602.nc')
LWR_depth_reg_201603 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201603.nc')
LWR_depth_reg_201604 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201604.nc')
LWR_depth_reg_201605 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201605.nc')
LWR_depth_reg_201606 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201606.nc')
LWR_depth_reg_201607 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201607.nc')
LWR_depth_reg_201608 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201608.nc')
LWR_depth_reg_201609 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201609.nc')
LWR_depth_reg_201610 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201610.nc')
LWR_depth_reg_201611 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201611.nc')
LWR_depth_reg_201612 = nc_to_startfinal_points('behavior_test/depth_reg/LWR_201612.nc')

lwr_depth_reg = LWR_depth_reg_201601 + LWR_depth_reg_201602 + LWR_depth_reg_201603 + LWR_depth_reg_201604 + LWR_depth_reg_201605 + LWR_depth_reg_201606 + LWR_depth_reg_201607 + LWR_depth_reg_201608 + LWR_depth_reg_201609 + LWR_depth_reg_201610 + LWR_depth_reg_201611 + LWR_depth_reg_201612



lwrx = [lwr_passive[i][1].x for i in range(len(lwr_passive))]
lwry = [lwr_passive[i][1].y for i in range(len(lwr_passive))]
lon_edges = np.linspace(164, 184, 201)
lat_edges = np.linspace(-52, -31, 211)
lwrH, _, _ = np.histogram2d(lwry, lwrx, [lat_edges, lon_edges]) #density = True
lwrx2 = [lwr_depth_reg[i][1].x for i in range(len(lwr_depth_reg))]
lwry2 = [lwr_depth_reg[i][1].y for i in range(len(lwr_depth_reg))]
lwrH2, _, _ = np.histogram2d(lwry2, lwrx2, [lat_edges, lon_edges]) #density = True

lwrHdiff = lwrH2-lwrH
#Hdiff[Hdiff==0]=None
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)


fig, axes = plt.subplots(1,3)
axes = axes.flatten()
m = Basemap(resolution= 'h', llcrnrlon = 169, llcrnrlat = -44, urcrnrlon = 177, urcrnrlat = -35, area_thresh=10000., ax = axes[0])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
ph = axes[0].pcolormesh(xs, ys, lwrH, norm=colors.LogNorm(), cmap = 'jet') # 
#ph.set_clim(0, 300)
fig.colorbar(ph, ax = axes[0], ticks = [50, 100, 500, 1000, 5000], format = '%.0f')
axes[0].set_title('Passive particles')
axes[0].plot(171.95, -41.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)


m = Basemap(resolution= 'h', llcrnrlon = 169, llcrnrlat = -44, urcrnrlon = 177, urcrnrlat = -35, area_thresh=10000., ax = axes[1])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
ph2 = axes[1].pcolormesh(xs, ys, lwrH2, norm=colors.LogNorm(), cmap = 'jet') #
#ph2.set_clim(0, 300)
fig.colorbar(ph2, ax = axes[1], ticks = [50, 100, 500, 1000, 5000], format = '%.0f')
axes[1].set_title('Depth regulation')
axes[1].plot(171.95, -41.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)


m = Basemap(resolution= 'h', llcrnrlon = 169, llcrnrlat = -44, urcrnrlon = 177, urcrnrlat = -35, area_thresh=10000., ax = axes[2])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdiff = axes[2].pcolormesh(xs, ys, lwrHdiff, cmap = 'seismic') #norm=colors.LogNorm()
pdiff.set_clim(-200, 200)
fig.colorbar(pdiff, ax = axes[2])
axes[2].set_title('Difference')
axes[2].plot(171.95, -41.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)

plt.show()

REI_passive_201601 = nc_to_startfinal_points('behavior_test/passive/REI_201601.nc')
REI_passive_201602 = nc_to_startfinal_points('behavior_test/passive/REI_201602.nc')
REI_passive_201603 = nc_to_startfinal_points('behavior_test/passive/REI_201603.nc')
REI_passive_201604 = nc_to_startfinal_points('behavior_test/passive/REI_201604.nc')
REI_passive_201605 = nc_to_startfinal_points('behavior_test/passive/REI_201605.nc')
REI_passive_201606 = nc_to_startfinal_points('behavior_test/passive/REI_201606.nc')
REI_passive_201607 = nc_to_startfinal_points('behavior_test/passive/REI_201607.nc')
REI_passive_201608 = nc_to_startfinal_points('behavior_test/passive/REI_201608.nc')
REI_passive_201609 = nc_to_startfinal_points('behavior_test/passive/REI_201609.nc')
REI_passive_201610 = nc_to_startfinal_points('behavior_test/passive/REI_201610.nc')
REI_passive_201611 = nc_to_startfinal_points('behavior_test/passive/REI_201611.nc')
REI_passive_201612 = nc_to_startfinal_points('behavior_test/passive/REI_201612.nc')

rei_passive = REI_passive_201601 + REI_passive_201602 + REI_passive_201603 + REI_passive_201604 + REI_passive_201605 + REI_passive_201606 + REI_passive_201607 + REI_passive_201608 + REI_passive_201609 + REI_passive_201610 + REI_passive_201611 + REI_passive_201612

REI_depth_reg_201601 = nc_to_startfinal_points('behavior_test/depth_reg/REI_201601.nc')
REI_depth_reg_201602 = nc_to_startfinal_points('behavior_test/depth_reg/REI_201602.nc')
REI_depth_reg_201603 = nc_to_startfinal_points('behavior_test/depth_reg/REI_201603.nc')
REI_depth_reg_201604 = nc_to_startfinal_points('behavior_test/depth_reg/REI_201604.nc')
REI_depth_reg_201605 = nc_to_startfinal_points('behavior_test/depth_reg/REI_201605.nc')
REI_depth_reg_201606 = nc_to_startfinal_points('behavior_test/depth_reg/REI_201606.nc')
REI_depth_reg_201607 = nc_to_startfinal_points('behavior_test/depth_reg/REI_201607.nc')
REI_depth_reg_201608 = nc_to_startfinal_points('behavior_test/depth_reg/REI_201608.nc')
REI_depth_reg_201609 = nc_to_startfinal_points('behavior_test/depth_reg/REI_201609.nc')
REI_depth_reg_201610 = nc_to_startfinal_points('behavior_test/depth_reg/REI_201610.nc')
REI_depth_reg_201611 = nc_to_startfinal_points('behavior_test/depth_reg/REI_201611.nc')
REI_depth_reg_201612 = nc_to_startfinal_points('behavior_test/depth_reg/REI_201612.nc')

rei_depth_reg = REI_depth_reg_201601 + REI_depth_reg_201602 + REI_depth_reg_201603 + REI_depth_reg_201604 + REI_depth_reg_201605 + REI_depth_reg_201606 + REI_depth_reg_201607 + REI_depth_reg_201608 + REI_depth_reg_201609 + REI_depth_reg_201610 + REI_depth_reg_201611 + REI_depth_reg_201612

DUN_passive_201601 = nc_to_startfinal_points('behavior_test/passive/DUN_201601.nc')
DUN_passive_201602 = nc_to_startfinal_points('behavior_test/passive/DUN_201602.nc')
DUN_passive_201603 = nc_to_startfinal_points('behavior_test/passive/DUN_201603.nc')
DUN_passive_201604 = nc_to_startfinal_points('behavior_test/passive/DUN_201604.nc')
DUN_passive_201605 = nc_to_startfinal_points('behavior_test/passive/DUN_201605.nc')
DUN_passive_201606 = nc_to_startfinal_points('behavior_test/passive/DUN_201606.nc')
DUN_passive_201607 = nc_to_startfinal_points('behavior_test/passive/DUN_201607.nc')
DUN_passive_201608 = nc_to_startfinal_points('behavior_test/passive/DUN_201608.nc')
DUN_passive_201609 = nc_to_startfinal_points('behavior_test/passive/DUN_201609.nc')
DUN_passive_201610 = nc_to_startfinal_points('behavior_test/passive/DUN_201610.nc')
DUN_passive_201611 = nc_to_startfinal_points('behavior_test/passive/DUN_201611.nc')
DUN_passive_201612 = nc_to_startfinal_points('behavior_test/passive/DUN_201612.nc')

dun_passive = DUN_passive_201601 + DUN_passive_201602 + DUN_passive_201603 + DUN_passive_201604 + DUN_passive_201605 + DUN_passive_201606 + DUN_passive_201607 + DUN_passive_201608 + DUN_passive_201609 + DUN_passive_201610 + DUN_passive_201611 + DUN_passive_201612

DUN_depth_reg_201601 = nc_to_startfinal_points('behavior_test/depth_reg/DUN_201601.nc')
DUN_depth_reg_201602 = nc_to_startfinal_points('behavior_test/depth_reg/DUN_201602.nc')
DUN_depth_reg_201603 = nc_to_startfinal_points('behavior_test/depth_reg/DUN_201603.nc')
DUN_depth_reg_201604 = nc_to_startfinal_points('behavior_test/depth_reg/DUN_201604.nc')
DUN_depth_reg_201605 = nc_to_startfinal_points('behavior_test/depth_reg/DUN_201605.nc')
DUN_depth_reg_201606 = nc_to_startfinal_points('behavior_test/depth_reg/DUN_201606.nc')
DUN_depth_reg_201607 = nc_to_startfinal_points('behavior_test/depth_reg/DUN_201607.nc')
DUN_depth_reg_201608 = nc_to_startfinal_points('behavior_test/depth_reg/DUN_201608.nc')
DUN_depth_reg_201609 = nc_to_startfinal_points('behavior_test/depth_reg/DUN_201609.nc')
DUN_depth_reg_201610 = nc_to_startfinal_points('behavior_test/depth_reg/DUN_201610.nc')
DUN_depth_reg_201611 = nc_to_startfinal_points('behavior_test/depth_reg/DUN_201611.nc')
DUN_depth_reg_201612 = nc_to_startfinal_points('behavior_test/depth_reg/DUN_201612.nc')

dun_depth_reg = DUN_depth_reg_201601 + DUN_depth_reg_201602 + DUN_depth_reg_201603 + DUN_depth_reg_201604 + DUN_depth_reg_201605 + DUN_depth_reg_201606 + DUN_depth_reg_201607 + DUN_depth_reg_201608 + DUN_depth_reg_201609 + DUN_depth_reg_201610 + DUN_depth_reg_201611 + DUN_depth_reg_201612

reix = [rei_passive[i][1].x for i in range(len(rei_passive))]
reiy = [rei_passive[i][1].y for i in range(len(rei_passive))]
lon_edges = np.linspace(164, 184, 201)
lat_edges = np.linspace(-52, -31, 211)
reiH, _, _ = np.histogram2d(reiy, reix, [lat_edges, lon_edges]) #density = True
reix2 = [rei_depth_reg[i][1].x for i in range(len(rei_depth_reg))]
reiy2 = [rei_depth_reg[i][1].y for i in range(len(rei_depth_reg))]
reiH2, _, _ = np.histogram2d(reiy2, reix2, [lat_edges, lon_edges]) #density = True

reiHdiff = reiH-reiH2
#Hdiff[Hdiff==0]=None
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)

from matplotlib.ticker import LogFormatter

fig, axes = plt.subplots(1,3)
axes = axes.flatten()
m = Basemap(resolution= 'h', llcrnrlon = 170, llcrnrlat = -40, urcrnrlon = 178, urcrnrlat = -32, area_thresh=10000., ax = axes[0])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
ph = axes[0].pcolormesh(xs, ys, reiH, norm=colors.LogNorm(), cmap = 'jet') # 
formatter = LogFormatter(10, labelOnlyBase=False) 
#ph.set_clim(0, 300)
fig.colorbar(ph, ax = axes[0], ticks = [50, 100, 500, 1000, 5000], format = '%.0f')
axes[0].set_title('Passive particles')
axes[0].plot(172.65, -34.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)


m = Basemap(resolution= 'h', llcrnrlon = 170, llcrnrlat = -40, urcrnrlon = 178, urcrnrlat = -32, area_thresh=10000., ax = axes[1])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
ph2 = axes[1].pcolormesh(xs, ys, reiH2, norm=colors.LogNorm(), cmap = 'jet') #
#ph2.set_clim(0, 300)
fig.colorbar(ph2, ax = axes[1], ticks = [50, 100, 500, 1000, 5000], format = '%.0f')
axes[1].set_title('Depth regulation')
axes[1].plot(172.65, -34.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)


m = Basemap(resolution= 'h', llcrnrlon = 170, llcrnrlat = -40, urcrnrlon = 178, urcrnrlat = -32, area_thresh=10000., ax = axes[2])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdiff = axes[2].pcolormesh(xs, ys, reiHdiff, cmap = 'seismic') #norm=colors.LogNorm()
pdiff.set_clim(-200, 200)
fig.colorbar(pdiff, ax = axes[2])
axes[2].set_title('Difference')
axes[2].plot(172.65, -34.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)
plt.show()





dunx = [dun_passive[i][1].x for i in range(len(dun_passive))]
duny = [dun_passive[i][1].y for i in range(len(dun_passive))]
lon_edges = np.linspace(164, 184, 201)
lat_edges = np.linspace(-52, -31, 211)
dunH, _, _ = np.histogram2d(duny, dunx, [lat_edges, lon_edges]) #density = True
dunx2 = [dun_depth_reg[i][1].x for i in range(len(dun_depth_reg))]
duny2 = [dun_depth_reg[i][1].y for i in range(len(dun_depth_reg))]
dunH2, _, _ = np.histogram2d(duny2, dunx2, [lat_edges, lon_edges]) #density = True

dunHdiff = dunH-dunH2
#Hdiff[Hdiff==0]=None
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)

from matplotlib.ticker import LogFormatter

fig, axes = plt.subplots(1,3)
axes = axes.flatten()
m = Basemap(resolution= 'h', llcrnrlon = 170, llcrnrlat = -47, urcrnrlon = 178, urcrnrlat = -40, area_thresh=10000., ax = axes[0])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
ph = axes[0].pcolormesh(xs, ys, dunH, norm=colors.LogNorm(), cmap = 'jet') # 
formatter = LogFormatter(10, labelOnlyBase=False) 
#ph.set_clim(0, 300)
fig.colorbar(ph, ax = axes[0], ticks = [50, 100, 500, 1000, 5000], format = '%.0f')
axes[0].set_title('Passive particles')
axes[0].plot(170.85, -45.85, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)

m = Basemap(resolution= 'h', llcrnrlon = 170, llcrnrlat = -47, urcrnrlon = 178, urcrnrlat = -40, area_thresh=10000., ax = axes[1])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
ph2 = axes[1].pcolormesh(xs, ys, dunH2, norm=colors.LogNorm(), cmap = 'jet') #
#ph2.set_clim(0, 300)
fig.colorbar(ph2, ax = axes[1], ticks = [50, 100, 500, 1000, 5000], format = '%.0f')
axes[1].set_title('Depth regulation')
axes[1].plot(170.85, -45.85, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)

m = Basemap(resolution= 'h', llcrnrlon = 170, llcrnrlat = -47, urcrnrlon = 178, urcrnrlat = -40, area_thresh=10000., ax = axes[2])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdiff = axes[2].pcolormesh(xs, ys, dunHdiff, cmap = 'seismic') #norm=colors.LogNorm()
pdiff.set_clim(-200, 200)
fig.colorbar(pdiff, ax = axes[2])
axes[2].set_title('Difference')
axes[2].plot(170.85, -45.85, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)
plt.show()










LWR_passive_201601 = nc_to_startfinal_points('behavior_test/passive/all_settlement/LWR_201601.nc')
LWR_passive_201602 = nc_to_startfinal_points('behavior_test/passive/all_settlement/LWR_201602.nc')
LWR_passive_201603 = nc_to_startfinal_points('behavior_test/passive/all_settlement/LWR_201603.nc')
LWR_passive_201604 = nc_to_startfinal_points('behavior_test/passive/all_settlement/LWR_201604.nc')
LWR_passive_201605 = nc_to_startfinal_points('behavior_test/passive/all_settlement/LWR_201605.nc')
LWR_passive_201606 = nc_to_startfinal_points('behavior_test/passive/all_settlement/LWR_201606.nc')
LWR_passive_201607 = nc_to_startfinal_points('behavior_test/passive/all_settlement/LWR_201607.nc')
LWR_passive_201608 = nc_to_startfinal_points('behavior_test/passive/all_settlement/LWR_201608.nc')
LWR_passive_201609 = nc_to_startfinal_points('behavior_test/passive/all_settlement/LWR_201609.nc')
LWR_passive_201610 = nc_to_startfinal_points('behavior_test/passive/all_settlement/LWR_201610.nc')
LWR_passive_201611 = nc_to_startfinal_points('behavior_test/passive/all_settlement/LWR_201611.nc')
LWR_passive_201612 = nc_to_startfinal_points('behavior_test/passive/all_settlement/LWR_201612.nc')

lwr_passive = LWR_passive_201601 + LWR_passive_201602 + LWR_passive_201603 + LWR_passive_201604 + LWR_passive_201605 + LWR_passive_201606 + LWR_passive_201607 + LWR_passive_201608 + LWR_passive_201609 + LWR_passive_201610 + LWR_passive_201611 + LWR_passive_201612
lwr_passives = [LWR_passive_201601, LWR_passive_201602, LWR_passive_201603, LWR_passive_201604, LWR_passive_201605, LWR_passive_201606, LWR_passive_201607, LWR_passive_201608, LWR_passive_201609, LWR_passive_201610, LWR_passive_201611, LWR_passive_201612]

LWR_depth_reg_201601 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/LWR_201601.nc')
LWR_depth_reg_201602 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/LWR_201602.nc')
LWR_depth_reg_201603 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/LWR_201603.nc')
LWR_depth_reg_201604 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/LWR_201604.nc')
LWR_depth_reg_201605 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/LWR_201605.nc')
LWR_depth_reg_201606 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/LWR_201606.nc')
LWR_depth_reg_201607 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/LWR_201607.nc')
LWR_depth_reg_201608 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/LWR_201608.nc')
LWR_depth_reg_201609 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/LWR_201609.nc')
LWR_depth_reg_201610 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/LWR_201610.nc')
LWR_depth_reg_201611 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/LWR_201611.nc')
LWR_depth_reg_201612 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/LWR_201612.nc')

lwr_depth_reg = LWR_depth_reg_201601 + LWR_depth_reg_201602 + LWR_depth_reg_201603 + LWR_depth_reg_201604 + LWR_depth_reg_201605 + LWR_depth_reg_201606 + LWR_depth_reg_201607 + LWR_depth_reg_201608 + LWR_depth_reg_201609 + LWR_depth_reg_201610 + LWR_depth_reg_201611 + LWR_depth_reg_201612
lwr_depth_regs = [LWR_depth_reg_201601, LWR_depth_reg_201602, LWR_depth_reg_201603, LWR_depth_reg_201604, LWR_depth_reg_201605, LWR_depth_reg_201606, LWR_depth_reg_201607, LWR_depth_reg_201608, LWR_depth_reg_201609, LWR_depth_reg_201610, LWR_depth_reg_201611, LWR_depth_reg_201612]

LWR_depth_reg_5to50_201601 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/LWR_201601.nc')
LWR_depth_reg_5to50_201602 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/LWR_201602.nc')
LWR_depth_reg_5to50_201603 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/LWR_201603.nc')
LWR_depth_reg_5to50_201604 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/LWR_201604.nc')
LWR_depth_reg_5to50_201605 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/LWR_201605.nc')
LWR_depth_reg_5to50_201606 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/LWR_201606.nc')
LWR_depth_reg_5to50_201607 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/LWR_201607.nc')
LWR_depth_reg_5to50_201608 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/LWR_201608.nc')
LWR_depth_reg_5to50_201609 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/LWR_201609.nc')
LWR_depth_reg_5to50_201610 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/LWR_201610.nc')
LWR_depth_reg_5to50_201611 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/LWR_201611.nc')
LWR_depth_reg_5to50_201612 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/LWR_201612.nc')

lwr_depth_reg_5to50 = LWR_depth_reg_5to50_201601 + LWR_depth_reg_5to50_201602 + LWR_depth_reg_5to50_201603 + LWR_depth_reg_5to50_201604 + LWR_depth_reg_5to50_201605 + LWR_depth_reg_5to50_201606 + LWR_depth_reg_5to50_201607 + LWR_depth_reg_5to50_201608 + LWR_depth_reg_5to50_201609 + LWR_depth_reg_5to50_201610 + LWR_depth_reg_5to50_201611 + LWR_depth_reg_5to50_201612
lwr_depth_regs_5to50 = [LWR_depth_reg_5to50_201601, LWR_depth_reg_5to50_201602, LWR_depth_reg_5to50_201603, LWR_depth_reg_5to50_201604, LWR_depth_reg_5to50_201605, LWR_depth_reg_5to50_201606, LWR_depth_reg_5to50_201607, LWR_depth_reg_5to50_201608, LWR_depth_reg_5to50_201609, LWR_depth_reg_5to50_201610, LWR_depth_reg_5to50_201611, LWR_depth_reg_5to50_201612]

LWR_sniff_201601 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/LWR_201601.nc')
LWR_sniff_201602 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/LWR_201602.nc')
LWR_sniff_201603 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/LWR_201603.nc')
LWR_sniff_201604 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/LWR_201604.nc')
LWR_sniff_201605 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/LWR_201605.nc')
LWR_sniff_201606 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/LWR_201606.nc')
LWR_sniff_201607 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/LWR_201607.nc')
LWR_sniff_201608 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/LWR_201608.nc')
LWR_sniff_201609 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/LWR_201609.nc')
LWR_sniff_201610 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/LWR_201610.nc')
LWR_sniff_201611 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/LWR_201611.nc')
LWR_sniff_201612 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/LWR_201612.nc')

lwr_sniff = LWR_sniff_201601 + LWR_sniff_201602 + LWR_sniff_201603 + LWR_sniff_201604 + LWR_sniff_201605 + LWR_sniff_201606 + LWR_sniff_201607 + LWR_sniff_201608 + LWR_sniff_201609 + LWR_sniff_201610 + LWR_sniff_201611 + LWR_sniff_201612
lwr_sniffs = [LWR_sniff_201601, LWR_sniff_201602, LWR_sniff_201603, LWR_sniff_201604, LWR_sniff_201605, LWR_sniff_201606, LWR_sniff_201607, LWR_sniff_201608, LWR_sniff_201609, LWR_sniff_201610, LWR_sniff_201611, LWR_sniff_201612]


REI_passive_201601 = nc_to_startfinal_points('behavior_test/passive/all_settlement/REI_201601.nc')
REI_passive_201602 = nc_to_startfinal_points('behavior_test/passive/all_settlement/REI_201602.nc')
REI_passive_201603 = nc_to_startfinal_points('behavior_test/passive/all_settlement/REI_201603.nc')
REI_passive_201604 = nc_to_startfinal_points('behavior_test/passive/all_settlement/REI_201604.nc')
REI_passive_201605 = nc_to_startfinal_points('behavior_test/passive/all_settlement/REI_201605.nc')
REI_passive_201606 = nc_to_startfinal_points('behavior_test/passive/all_settlement/REI_201606.nc')
REI_passive_201607 = nc_to_startfinal_points('behavior_test/passive/all_settlement/REI_201607.nc')
REI_passive_201608 = nc_to_startfinal_points('behavior_test/passive/all_settlement/REI_201608.nc')
REI_passive_201609 = nc_to_startfinal_points('behavior_test/passive/all_settlement/REI_201609.nc')
REI_passive_201610 = nc_to_startfinal_points('behavior_test/passive/all_settlement/REI_201610.nc')
REI_passive_201611 = nc_to_startfinal_points('behavior_test/passive/all_settlement/REI_201611.nc')
REI_passive_201612 = nc_to_startfinal_points('behavior_test/passive/all_settlement/REI_201612.nc')

rei_passive = REI_passive_201601 + REI_passive_201602 + REI_passive_201603 + REI_passive_201604 + REI_passive_201605 + REI_passive_201606 + REI_passive_201607 + REI_passive_201608 + REI_passive_201609 + REI_passive_201610 + REI_passive_201611 + REI_passive_201612
rei_passives = [REI_passive_201601, REI_passive_201602, REI_passive_201603, REI_passive_201604, REI_passive_201605, REI_passive_201606, REI_passive_201607, REI_passive_201608, REI_passive_201609, REI_passive_201610, REI_passive_201611, REI_passive_201612]

REI_depth_reg_201601 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/REI_201601.nc')
REI_depth_reg_201602 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/REI_201602.nc')
REI_depth_reg_201603 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/REI_201603.nc')
REI_depth_reg_201604 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/REI_201604.nc')
REI_depth_reg_201605 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/REI_201605.nc')
REI_depth_reg_201606 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/REI_201606.nc')
REI_depth_reg_201607 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/REI_201607.nc')
REI_depth_reg_201608 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/REI_201608.nc')
REI_depth_reg_201609 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/REI_201609.nc')
REI_depth_reg_201610 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/REI_201610.nc')
REI_depth_reg_201611 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/REI_201611.nc')
REI_depth_reg_201612 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/REI_201612.nc')

rei_depth_reg = REI_depth_reg_201601 + REI_depth_reg_201602 + REI_depth_reg_201603 + REI_depth_reg_201604 + REI_depth_reg_201605 + REI_depth_reg_201606 + REI_depth_reg_201607 + REI_depth_reg_201608 + REI_depth_reg_201609 + REI_depth_reg_201610 + REI_depth_reg_201611 + REI_depth_reg_201612
rei_depth_regs = [REI_depth_reg_201601, REI_depth_reg_201602, REI_depth_reg_201603, REI_depth_reg_201604, REI_depth_reg_201605, REI_depth_reg_201606, REI_depth_reg_201607, REI_depth_reg_201608, REI_depth_reg_201609, REI_depth_reg_201610, REI_depth_reg_201611, REI_depth_reg_201612]

REI_depth_reg_5to50_201601 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/REI_201601.nc')
REI_depth_reg_5to50_201602 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/REI_201602.nc')
REI_depth_reg_5to50_201603 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/REI_201603.nc')
REI_depth_reg_5to50_201604 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/REI_201604.nc')
REI_depth_reg_5to50_201605 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/REI_201605.nc')
REI_depth_reg_5to50_201606 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/REI_201606.nc')
REI_depth_reg_5to50_201607 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/REI_201607.nc')
REI_depth_reg_5to50_201608 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/REI_201608.nc')
REI_depth_reg_5to50_201609 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/REI_201609.nc')
REI_depth_reg_5to50_201610 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/REI_201610.nc')
REI_depth_reg_5to50_201611 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/REI_201611.nc')
REI_depth_reg_5to50_201612 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/REI_201612.nc')

rei_depth_reg_5to50 = REI_depth_reg_5to50_201601 + REI_depth_reg_5to50_201602 + REI_depth_reg_5to50_201603 + REI_depth_reg_5to50_201604 + REI_depth_reg_5to50_201605 + REI_depth_reg_5to50_201606 + REI_depth_reg_5to50_201607 + REI_depth_reg_5to50_201608 + REI_depth_reg_5to50_201609 + REI_depth_reg_5to50_201610 + REI_depth_reg_5to50_201611 + REI_depth_reg_5to50_201612
rei_depth_regs_5to50 = [REI_depth_reg_5to50_201601, REI_depth_reg_5to50_201602, REI_depth_reg_5to50_201603, REI_depth_reg_5to50_201604, REI_depth_reg_5to50_201605, REI_depth_reg_5to50_201606, REI_depth_reg_5to50_201607, REI_depth_reg_5to50_201608, REI_depth_reg_5to50_201609, REI_depth_reg_5to50_201610, REI_depth_reg_5to50_201611, REI_depth_reg_5to50_201612]

REI_sniff_201601 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/REI_201601.nc')
REI_sniff_201602 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/REI_201602.nc')
REI_sniff_201603 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/REI_201603.nc')
REI_sniff_201604 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/REI_201604.nc')
REI_sniff_201605 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/REI_201605.nc')
REI_sniff_201606 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/REI_201606.nc')
REI_sniff_201607 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/REI_201607.nc')
REI_sniff_201608 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/REI_201608.nc')
REI_sniff_201609 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/REI_201609.nc')
REI_sniff_201610 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/REI_201610.nc')
REI_sniff_201611 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/REI_201611.nc')
REI_sniff_201612 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/REI_201612.nc')

rei_sniff = REI_sniff_201601 + REI_sniff_201602 + REI_sniff_201603 + REI_sniff_201604 + REI_sniff_201605 + REI_sniff_201606 + REI_sniff_201607 + REI_sniff_201608 + REI_sniff_201609 + REI_sniff_201610 + REI_sniff_201611 + REI_sniff_201612
rei_sniffs = [REI_sniff_201601, REI_sniff_201602, REI_sniff_201603, REI_sniff_201604, REI_sniff_201605, REI_sniff_201606, REI_sniff_201607, REI_sniff_201608, REI_sniff_201609, REI_sniff_201610, REI_sniff_201611, REI_sniff_201612]

DUN_passive_201601 = nc_to_startfinal_points('behavior_test/passive/all_settlement/DUN_201601.nc')
DUN_passive_201602 = nc_to_startfinal_points('behavior_test/passive/all_settlement/DUN_201602.nc')
DUN_passive_201603 = nc_to_startfinal_points('behavior_test/passive/all_settlement/DUN_201603.nc')
DUN_passive_201604 = nc_to_startfinal_points('behavior_test/passive/all_settlement/DUN_201604.nc')
DUN_passive_201605 = nc_to_startfinal_points('behavior_test/passive/all_settlement/DUN_201605.nc')
DUN_passive_201606 = nc_to_startfinal_points('behavior_test/passive/all_settlement/DUN_201606.nc')
DUN_passive_201607 = nc_to_startfinal_points('behavior_test/passive/all_settlement/DUN_201607.nc')
DUN_passive_201608 = nc_to_startfinal_points('behavior_test/passive/all_settlement/DUN_201608.nc')
DUN_passive_201609 = nc_to_startfinal_points('behavior_test/passive/all_settlement/DUN_201609.nc')
DUN_passive_201610 = nc_to_startfinal_points('behavior_test/passive/all_settlement/DUN_201610.nc')
DUN_passive_201611 = nc_to_startfinal_points('behavior_test/passive/all_settlement/DUN_201611.nc')
DUN_passive_201612 = nc_to_startfinal_points('behavior_test/passive/all_settlement/DUN_201612.nc')

dun_passive = DUN_passive_201601 + DUN_passive_201602 + DUN_passive_201603 + DUN_passive_201604 + DUN_passive_201605 + DUN_passive_201606 + DUN_passive_201607 + DUN_passive_201608 + DUN_passive_201609 + DUN_passive_201610 + DUN_passive_201611 + DUN_passive_201612
dun_passives = [DUN_passive_201601, DUN_passive_201602, DUN_passive_201603, DUN_passive_201604, DUN_passive_201605, DUN_passive_201606, DUN_passive_201607, DUN_passive_201608, DUN_passive_201609, DUN_passive_201610, DUN_passive_201611, DUN_passive_201612]

DUN_depth_reg_201601 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/DUN_201601.nc')
DUN_depth_reg_201602 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/DUN_201602.nc')
DUN_depth_reg_201603 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/DUN_201603.nc')
DUN_depth_reg_201604 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/DUN_201604.nc')
DUN_depth_reg_201605 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/DUN_201605.nc')
DUN_depth_reg_201606 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/DUN_201606.nc')
DUN_depth_reg_201607 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/DUN_201607.nc')
DUN_depth_reg_201608 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/DUN_201608.nc')
DUN_depth_reg_201609 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/DUN_201609.nc')
DUN_depth_reg_201610 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/DUN_201610.nc')
DUN_depth_reg_201611 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/DUN_201611.nc')
DUN_depth_reg_201612 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/DUN_201612.nc')

dun_depth_reg = DUN_depth_reg_201601 + DUN_depth_reg_201602 + DUN_depth_reg_201603 + DUN_depth_reg_201604 + DUN_depth_reg_201605 + DUN_depth_reg_201606 + DUN_depth_reg_201607 + DUN_depth_reg_201608 + DUN_depth_reg_201609 + DUN_depth_reg_201610 + DUN_depth_reg_201611 + DUN_depth_reg_201612
dun_depth_regs = [DUN_depth_reg_201601, DUN_depth_reg_201602, DUN_depth_reg_201603, DUN_depth_reg_201604, DUN_depth_reg_201605, DUN_depth_reg_201606, DUN_depth_reg_201607, DUN_depth_reg_201608, DUN_depth_reg_201609, DUN_depth_reg_201610, DUN_depth_reg_201611, DUN_depth_reg_201612]

DUN_depth_reg_5to50_201601 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/DUN_201601.nc')
DUN_depth_reg_5to50_201602 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/DUN_201602.nc')
DUN_depth_reg_5to50_201603 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/DUN_201603.nc')
DUN_depth_reg_5to50_201604 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/DUN_201604.nc')
DUN_depth_reg_5to50_201605 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/DUN_201605.nc')
DUN_depth_reg_5to50_201606 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/DUN_201606.nc')
DUN_depth_reg_5to50_201607 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/DUN_201607.nc')
DUN_depth_reg_5to50_201608 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/DUN_201608.nc')
DUN_depth_reg_5to50_201609 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/DUN_201609.nc')
DUN_depth_reg_5to50_201610 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/DUN_201610.nc')
DUN_depth_reg_5to50_201611 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/DUN_201611.nc')
DUN_depth_reg_5to50_201612 = nc_to_startfinal_points('behavior_test/depth_reg/all_settlement/5to50/DUN_201612.nc')

dun_depth_reg_5to50 = DUN_depth_reg_5to50_201601 + DUN_depth_reg_5to50_201602 + DUN_depth_reg_5to50_201603 + DUN_depth_reg_5to50_201604 + DUN_depth_reg_5to50_201605 + DUN_depth_reg_5to50_201606 + DUN_depth_reg_5to50_201607 + DUN_depth_reg_5to50_201608 + DUN_depth_reg_5to50_201609 + DUN_depth_reg_5to50_201610 + DUN_depth_reg_5to50_201611 + DUN_depth_reg_5to50_201612
dun_depth_regs_5to50 = [DUN_depth_reg_5to50_201601, DUN_depth_reg_5to50_201602, DUN_depth_reg_5to50_201603, DUN_depth_reg_5to50_201604, DUN_depth_reg_5to50_201605, DUN_depth_reg_5to50_201606, DUN_depth_reg_5to50_201607, DUN_depth_reg_5to50_201608, DUN_depth_reg_5to50_201609, DUN_depth_reg_5to50_201610, DUN_depth_reg_5to50_201611, DUN_depth_reg_5to50_201612]

DUN_sniff_201601 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/DUN_201601.nc')
DUN_sniff_201602 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/DUN_201602.nc')
DUN_sniff_201603 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/DUN_201603.nc')
DUN_sniff_201604 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/DUN_201604.nc')
DUN_sniff_201605 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/DUN_201605.nc')
DUN_sniff_201606 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/DUN_201606.nc')
DUN_sniff_201607 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/DUN_201607.nc')
DUN_sniff_201608 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/DUN_201608.nc')
DUN_sniff_201609 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/DUN_201609.nc')
DUN_sniff_201610 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/DUN_201610.nc')
DUN_sniff_201611 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/DUN_201611.nc')
DUN_sniff_201612 = nc_to_startfinal_points('behavior_test/sniff/all_settlement/DUN_201612.nc')

dun_sniff = DUN_sniff_201601 + DUN_sniff_201602 + DUN_sniff_201603 + DUN_sniff_201604 + DUN_sniff_201605 + DUN_sniff_201606 + DUN_sniff_201607 + DUN_sniff_201608 + DUN_sniff_201609 + DUN_sniff_201610 + DUN_sniff_201611 + DUN_sniff_201612
dun_sniffs = [DUN_sniff_201601, DUN_sniff_201602, DUN_sniff_201603, DUN_sniff_201604, DUN_sniff_201605, DUN_sniff_201606, DUN_sniff_201607, DUN_sniff_201608, DUN_sniff_201609, DUN_sniff_201610, DUN_sniff_201611, DUN_sniff_201612]


[len([j for j in i if j[2] == 'home_sweet_home']) for i in lwr_depth_regs]

[len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in lwr_passives]
[len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in lwr_depth_regs]

plt.plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in lwr_passives], 'b', label = 'LWR passive')
plt.plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in lwr_depth_regs], 'b--', label = 'LWR depth regulation (5 to 25 m)')
plt.plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in lwr_depth_regs_5to50], 'b-.', label = 'LWR depth regulation (5 to 50 m)')
plt.plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in lwr_sniffs], 'b:', label = 'LWR Super sniffing')

plt.plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in rei_passives], 'r', label = 'REI passive')
plt.plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in rei_depth_regs], 'r--', label = 'REI depth regulation (5 to 25 m')
plt.plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in lwr_depth_regs_5to50], 'r-.', label = 'REI depth regulation (5 to 50 m)')
plt.plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in lwr_sniffs], 'r:', label = 'REI Super sniffing')

plt.plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in dun_passives], 'k', label = 'DUN passive')
plt.plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in dun_depth_regs], 'k--', label = 'DUN depth regulation (5 to 25 m)')
plt.plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in lwr_depth_regs_5to50], 'k-.', label = 'DUN depth regulation (5 to 50 m)')
plt.plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in lwr_sniffs], 'k:', label = 'DUN Super sniffing')

plt.ylabel('Percent successful settlement')
plt.xlabel('Month (2016)')
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(loc = 'upper right')
plt.show()

fig, axes = plt.subplots(3,1)
axes = axes.flatten()
axes[0].plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in lwr_passives], 'b')
axes[0].plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in lwr_depth_regs], 'b--')
axes[0].plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in lwr_depth_regs_5to50], 'b-.')
axes[0].plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in lwr_sniffs], 'b:')
axes[0].set_xticks(range(12))
axes[0].set_xticklabels([])

axes[1].plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in rei_passives], 'r')
axes[1].plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in rei_depth_regs], 'r--')
axes[1].plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in rei_depth_regs_5to50], 'r-.')
axes[1].plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in rei_sniffs], 'r:')
axes[1].set_xticks(range(12))
axes[1].set_xticklabels([])

axes[2].plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in dun_passives], 'k')
axes[2].plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in dun_depth_regs], 'k--')
axes[2].plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in dun_depth_regs_5to50], 'k-.')
axes[2].plot(range(12), [len([1 for j in i if j[2] =='home_sweet_home'])/len(i) for i in dun_sniffs], 'k:')
axes[2].set_xticks(range(12))
axes[2].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

fig.text(0.5, 0.04, 'Month (2016)', ha='center')
fig.text(0.04, 0.5, 'Percent successful settlement', va='center', rotation='vertical')
fig.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
fig.legend(custom_lines, ['LWR', 'REI', 'DUN', '', 'Passive', 'Depth regulation (5 to 25 m)', 'Depth regulation (5 to 50 m)', 'Super Sniffing'], loc = 'upper right', ncol = 2)
plt.show()



from matplotlib.lines import Line2D

custom_lines = [Line2D([0], [0], color = 'b'),
                Line2D([0], [0], color = 'r'),
                Line2D([0], [0], color = 'k'),
                Line2D([0], [0], linestyle = 'None'),
                Line2D([0], [0], color = 'k', linestyle = '-'),
                Line2D([0], [0], color = 'k', linestyle = '--'),
                Line2D([0], [0], color = 'k', linestyle = '-.'),
                Line2D([0], [0], color = 'k', linestyle = ':')]



lwrx = [lwr_passive[i][1].x for i in range(len(lwr_passive))]
lwry = [lwr_passive[i][1].y for i in range(len(lwr_passive))]
lon_edges = np.linspace(164, 184, 201)
lat_edges = np.linspace(-52, -31, 211)
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)
lwrH, _, _ = np.histogram2d(lwry, lwrx, [lat_edges, lon_edges]) #density = True
lwrx2 = [lwr_depth_reg[i][1].x for i in range(len(lwr_depth_reg))]
lwry2 = [lwr_depth_reg[i][1].y for i in range(len(lwr_depth_reg))]
lwrH2, _, _ = np.histogram2d(lwry2, lwrx2, [lat_edges, lon_edges]) #density = True
lwrx3 = [lwr_depth_reg_5to50[i][1].x for i in range(len(lwr_depth_reg_5to50))]
lwry3 = [lwr_depth_reg_5to50[i][1].y for i in range(len(lwr_depth_reg_5to50))]
lwrH3, _, _ = np.histogram2d(lwry3, lwrx3, [lat_edges, lon_edges]) #density = True
lwrx4 = [lwr_sniff[i][1].x for i in range(len(lwr_sniff))]
lwry4 = [lwr_sniff[i][1].y for i in range(len(lwr_sniff))]
lwrH4, _, _ = np.histogram2d(lwry4, lwrx4, [lat_edges, lon_edges]) #density = True


lwrH2diff = lwrH2-lwrH
lwrH3diff = lwrH3-lwrH
lwrH4diff = lwrH4-lwrH



fig, axes = plt.subplots(3,2)
axes = axes.flatten()
m = Basemap(resolution= 'h', llcrnrlon = 169, llcrnrlat = -44, urcrnrlon = 177, urcrnrlat = -35, area_thresh=10000., ax = axes[0])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
ph = axes[0].pcolormesh(xs, ys, lwrH, norm=colors.LogNorm(), cmap = 'jet') # 
#ph.set_clim(0, 300)
fig.colorbar(ph, ax = axes[0], ticks = [50, 100, 500, 1000, 5000], format = '%.0f')
axes[0].set_title('Passive particles')
axes[0].plot(171.95, -41.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)

axes[1].set_visible(False)

m = Basemap(resolution= 'h', llcrnrlon = 169, llcrnrlat = -44, urcrnrlon = 177, urcrnrlat = -35, area_thresh=10000., ax = axes[2])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
ph2 = axes[2].pcolormesh(xs, ys, lwrH2, norm=colors.LogNorm(), cmap = 'jet') #
#ph2.set_clim(0, 300)
fig.colorbar(ph2, ax = axes[2], ticks = [50, 100, 500, 1000, 5000], format = '%.0f')
axes[2].set_title('Depth regulation (10 to 25 m)')
axes[2].plot(171.95, -41.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)

m = Basemap(resolution= 'h', llcrnrlon = 169, llcrnrlat = -44, urcrnrlon = 177, urcrnrlat = -35, area_thresh=10000., ax = axes[3])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdiff2 = axes[3].pcolormesh(xs, ys, lwrH2diff, cmap = 'seismic') #norm=colors.LogNorm()
pdiff2.set_clim(-200, 200)
fig.colorbar(pdiff2, ax = axes[3])
axes[3].set_title('Difference')
axes[3].plot(171.95, -41.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)


m = Basemap(resolution= 'h', llcrnrlon = 169, llcrnrlat = -44, urcrnrlon = 177, urcrnrlat = -35, area_thresh=10000., ax = axes[4])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
ph3 = axes[4].pcolormesh(xs, ys, lwrH3, norm=colors.LogNorm(), cmap = 'jet') #
#ph2.set_clim(0, 300)
fig.colorbar(ph3, ax = axes[4], ticks = [50, 100, 500, 1000, 5000], format = '%.0f')
axes[4].set_title('Depth regulation (5 to 50 m)')
axes[4].plot(171.95, -41.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)


m = Basemap(resolution= 'h', llcrnrlon = 169, llcrnrlat = -44, urcrnrlon = 177, urcrnrlat = -35, area_thresh=10000., ax = axes[5])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdiff3 = axes[5].pcolormesh(xs, ys, lwrH3diff, cmap = 'seismic') #norm=colors.LogNorm()
pdiff3.set_clim(-200, 200)
fig.colorbar(pdiff3, ax = axes[5])
axes[5].set_title('Difference')
axes[5].plot(171.95, -41.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)
plt.show()



fig, axes = plt.subplots(1,3)
axes = axes.flatten()

m = Basemap(resolution= 'h', llcrnrlon = 169, llcrnrlat = -44, urcrnrlon = 177, urcrnrlat = -35, area_thresh=10000., ax = axes[0])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
ph = axes[0].pcolormesh(xs, ys, lwrH, norm=colors.LogNorm(), cmap = 'jet') # 
#ph.set_clim(0, 300)
fig.colorbar(ph, ax = axes[0], ticks = [50, 100, 500, 1000, 5000], format = '%.0f')
axes[0].set_title('Passive particles')
axes[0].plot(171.95, -41.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)

m = Basemap(resolution= 'h', llcrnrlon = 169, llcrnrlat = -44, urcrnrlon = 177, urcrnrlat = -35, area_thresh=10000., ax = axes[1])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
ph2 = axes[1].pcolormesh(xs, ys, lwrH4, norm=colors.LogNorm(), cmap = 'jet') #
#ph2.set_clim(0, 300)
fig.colorbar(ph2, ax = axes[1], ticks = [50, 100, 500, 1000, 5000], format = '%.0f')
axes[1].set_title('Depth regulation')
axes[1].plot(171.95, -41.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)


m = Basemap(resolution= 'h', llcrnrlon = 169, llcrnrlat = -44, urcrnrlon = 177, urcrnrlat = -35, area_thresh=10000., ax = axes[2])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdiff = axes[2].pcolormesh(xs, ys, lwrH4diff, cmap = 'seismic') #norm=colors.LogNorm()
pdiff.set_clim(-200, 200)
fig.colorbar(pdiff, ax = axes[2])
axes[2].set_title('Difference')
axes[2].plot(171.95, -41.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)



reix = [rei_passive[i][1].x for i in range(len(rei_passive))]
reiy = [rei_passive[i][1].y for i in range(len(rei_passive))]
lon_edges = np.linspace(164, 184, 201)
lat_edges = np.linspace(-52, -31, 211)
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)
reiH, _, _ = np.histogram2d(reiy, reix, [lat_edges, lon_edges]) #density = True
reix2 = [rei_depth_reg[i][1].x for i in range(len(rei_depth_reg))]
reiy2 = [rei_depth_reg[i][1].y for i in range(len(rei_depth_reg))]
reiH2, _, _ = np.histogram2d(reiy2, reix2, [lat_edges, lon_edges]) #density = True
reix3 = [rei_depth_reg_5to50[i][1].x for i in range(len(rei_depth_reg_5to50))]
reiy3 = [rei_depth_reg_5to50[i][1].y for i in range(len(rei_depth_reg_5to50))]
reiH3, _, _ = np.histogram2d(reiy3, reix3, [lat_edges, lon_edges]) #density = True

reiH2diff = reiH2-reiH
reiH3diff = reiH3-reiH



fig, axes = plt.subplots(3,2)
axes = axes.flatten()
m = Basemap(resolution= 'h', llcrnrlon = 170, llcrnrlat = -40, urcrnrlon = 178, urcrnrlat = -32, area_thresh=10000., ax = axes[0])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
ph = axes[0].pcolormesh(xs, ys, reiH, norm=colors.LogNorm(), cmap = 'jet') # 
#ph.set_clim(0, 300)
fig.colorbar(ph, ax = axes[0], ticks = [50, 100, 500, 1000, 5000], format = '%.0f')
axes[0].set_title('Passive particles')
axes[0].plot(172.65, -34.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)

axes[1].set_visible(False)

m = Basemap(resolution= 'h', llcrnrlon = 170, llcrnrlat = -40, urcrnrlon = 178, urcrnrlat = -32, area_thresh=10000., ax = axes[2])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
ph2 = axes[2].pcolormesh(xs, ys, reiH2, norm=colors.LogNorm(), cmap = 'jet') #
#ph2.set_clim(0, 300)
fig.colorbar(ph2, ax = axes[2], ticks = [50, 100, 500, 1000, 5000], format = '%.0f')
axes[2].set_title('Depth regulation (10 to 25 m)')
axes[2].plot(172.65, -34.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)

m = Basemap(resolution= 'h', llcrnrlon = 170, llcrnrlat = -40, urcrnrlon = 178, urcrnrlat = -32, area_thresh=10000., ax = axes[3])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdiff2 = axes[3].pcolormesh(xs, ys, reiH2diff, cmap = 'seismic') #norm=colors.LogNorm()
pdiff2.set_clim(-200, 200)
fig.colorbar(pdiff2, ax = axes[3])
axes[3].set_title('Difference')
axes[3].plot(172.65, -34.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)


m = Basemap(resolution= 'h', llcrnrlon = 170, llcrnrlat = -40, urcrnrlon = 178, urcrnrlat = -32, area_thresh=10000., ax = axes[4])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
ph3 = axes[4].pcolormesh(xs, ys, reiH3, norm=colors.LogNorm(), cmap = 'jet') #
#ph2.set_clim(0, 300)
fig.colorbar(ph3, ax = axes[4], ticks = [50, 100, 500, 1000, 5000], format = '%.0f')
axes[4].set_title('Depth regulation (5 to 50 m)')
axes[4].plot(172.65, -34.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)


m = Basemap(resolution= 'h', llcrnrlon = 170, llcrnrlat = -40, urcrnrlon = 178, urcrnrlat = -32, area_thresh=10000., ax = axes[5])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdiff3 = axes[5].pcolormesh(xs, ys, reiH3diff, cmap = 'seismic') #norm=colors.LogNorm()
pdiff3.set_clim(-200, 200)
fig.colorbar(pdiff3, ax = axes[5])
axes[5].set_title('Difference')
axes[5].plot(172.65, -34.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)
plt.show()





fig, axes = plt.subplots(3,2)
axes = axes.flatten()
m = Basemap(resolution= 'h', llcrnrlon = 170, llcrnrlat = -40, urcrnrlon = 178, urcrnrlat = -32, area_thresh=10000., ax = axes[0])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
ph = axes[0].pcolormesh(xs, ys, lwrH, norm=colors.LogNorm(), cmap = 'jet') # 
#ph.set_clim(0, 300)
fig.colorbar(ph, ax = axes[0], ticks = [50, 100, 500, 1000, 5000], format = '%.0f')
axes[0].set_title('Passive particles')
axes[0].plot(171.95, -41.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)

axes[1].set_visible(False)

m = Basemap(resolution= 'h', llcrnrlon = 170, llcrnrlat = -40, urcrnrlon = 178, urcrnrlat = -32, area_thresh=10000., ax = axes[2])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
ph2 = axes[2].pcolormesh(xs, ys, lwrH2, norm=colors.LogNorm(), cmap = 'jet') #
#ph2.set_clim(0, 300)
fig.colorbar(ph2, ax = axes[2], ticks = [50, 100, 500, 1000, 5000], format = '%.0f')
axes[2].set_title('Depth regulation (10 to 25 m)')
axes[2].plot(171.95, -41.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)

m = Basemap(resolution= 'h', llcrnrlon = 170, llcrnrlat = -40, urcrnrlon = 178, urcrnrlat = -32, area_thresh=10000., ax = axes[3])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdiff2 = axes[3].pcolormesh(xs, ys, lwrH2diff, cmap = 'seismic') #norm=colors.LogNorm()
pdiff2.set_clim(-200, 200)
fig.colorbar(pdiff2, ax = axes[3])
axes[3].set_title('Difference')
axes[3].plot(171.95, -41.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)


m = Basemap(resolution= 'h', llcrnrlon = 170, llcrnrlat = -40, urcrnrlon = 178, urcrnrlat = -32, area_thresh=10000., ax = axes[4])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
ph3 = axes[4].pcolormesh(xs, ys, lwrH3, norm=colors.LogNorm(), cmap = 'jet') #
#ph2.set_clim(0, 300)
fig.colorbar(ph3, ax = axes[4], ticks = [50, 100, 500, 1000, 5000], format = '%.0f')
axes[4].set_title('Depth regulation (5 to 50 m)')
axes[4].plot(171.95, -41.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)


m = Basemap(resolution= 'h', llcrnrlon = 170, llcrnrlat = -40, urcrnrlon = 178, urcrnrlat = -32, area_thresh=10000., ax = axes[5])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdiff3 = axes[5].pcolormesh(xs, ys, lwrH3diff, cmap = 'seismic') #norm=colors.LogNorm()
pdiff3.set_clim(-200, 200)
fig.colorbar(pdiff3, ax = axes[5])
axes[5].set_title('Difference')
axes[5].plot(171.95, -41.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)
plt.show()






fig, axes = plt.subplots(1,3)
axes = axes.flatten()
m = Basemap(resolution= 'h', llcrnrlon = 169, llcrnrlat = -44, urcrnrlon = 177, urcrnrlat = -35, area_thresh=10000., ax = axes[0])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
ph = axes[0].pcolormesh(xs, ys, lwrH, norm=colors.LogNorm(), cmap = 'jet') # 
#ph.set_clim(0, 300)
fig.colorbar(ph, ax = axes[0], ticks = [50, 100, 500, 1000, 5000], format = '%.0f')
axes[0].set_title('Passive particles')
axes[0].plot(171.95, -41.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)


m = Basemap(resolution= 'h', llcrnrlon = 169, llcrnrlat = -44, urcrnrlon = 177, urcrnrlat = -35, area_thresh=10000., ax = axes[1])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
ph2 = axes[1].pcolormesh(xs, ys, lwrH4, norm=colors.LogNorm(), cmap = 'jet') #
#ph2.set_clim(0, 300)
fig.colorbar(ph2, ax = axes[1], ticks = [50, 100, 500, 1000, 5000], format = '%.0f')
axes[1].set_title('Super Sniffing')
axes[1].plot(171.95, -41.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)


m = Basemap(resolution= 'h', llcrnrlon = 169, llcrnrlat = -44, urcrnrlon = 177, urcrnrlat = -35, area_thresh=10000., ax = axes[2])
m.drawcoastlines()
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='oldlace')
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdiff = axes[2].pcolormesh(xs, ys, lwrH4diff, cmap = 'seismic') #norm=colors.LogNorm()
pdiff.set_clim(-200, 200)
fig.colorbar(pdiff, ax = axes[2])
axes[2].set_title('Difference')
axes[2].plot(171.95, -41.35, 's', markerfacecolor= 'none', markeredgecolor = 'black', markeredgewidth = 2)

plt.show()





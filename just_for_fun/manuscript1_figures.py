#!/usr/bin/env python3

import os
import sys
import numpy as np
import netCDF4 as nc
import pandas as pd
import math
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from mpl_toolkits.basemap import Basemap
from matplotlib import colors

from eofs.standard import Eof


#Connectivity matrices

bigboy14 = np.loadtxt('bigboy_all_settlement.txt')
just14 = bigboy14[:, [30, 80, 81, 157, 173, 187, 199, 235, 247, 249, 271, 316, 432, 442]]
just14 = just14[[30, 80, 81, 157, 173, 187, 199, 235, 247, 249, 271, 316, 432, 442], :]
site_order14 = [8, 12, 5, 6, 7, 4, 13, 11, 10, 9, 3, 2, 1, 0]
just14_order = just14[site_order14]
just14_order = just14_order[:,site_order14]
site_labs14 = ['FIO', 'BGB', 'HSB', 'TIM', 'LWR', 'WEST', 'FLE', 'TAS', 'OPO', 'GOB', 'KAI', 'CAM', 'MAU', 'CAP']
site_labs = np.asarray(site_labs14)
site_labs = site_labs[site_order14]

df = pd.DataFrame(data = just14_order, index = site_labs, columns = site_labs)
num=np.sum(bigboy[30])
df_pct=(df/num)*100
df_log=np.log10(df_pct)
df_log[np.isneginf(df_log)] = -5

ax = sns.heatmap(df_log, cbar_kws={'label': '% Succesful migrants)'})#, annot=True, fmt='.0f')
cbar = ax.collections[0].colorbar
cbar.set_ticks([1,0,-1,-2,-3])
cbar.set_ticklabels([10,1,.1,.01,.001])
ax.invert_xaxis()
plt.xlabel("Destination Population")
plt.ylabel("Source Population")
#plt.title("Ocean Connectivity")
plt.tight_layout()
plt.show()



#PDFs for each site

pdf_OPO = np.zeros((801,761))
pdf_MAU = np.zeros((801,761))
pdf_WEST = np.zeros((801,761))
pdf_FLE = np.zeros((801,761))
pdf_TAS = np.zeros((801,761))
pdf_LWR = np.zeros((801,761))
pdf_CAP = np.zeros((801,761))
pdf_CAM = np.zeros((801,761))
pdf_KAI = np.zeros((801,761))
pdf_GOB = np.zeros((801,761))
pdf_TIM = np.zeros((801,761))
pdf_HSB = np.zeros((801,761))
pdf_BGB = np.zeros((801,761))
pdf_FIO = np.zeros((801,761))


pdfs = [pdf_OPO,pdf_MAU,pdf_WEST,pdf_FLE,pdf_TAS,pdf_LWR,pdf_CAP,pdf_CAM,pdf_KAI,pdf_GOB,pdf_TIM,pdf_HSB,pdf_BGB,pdf_FIO]
sites = ['OPO','MAU','WEST','FLE','TAS','LWR','CAP','CAM','KAI','GOB','TIM','HSB','BGB','FIO']

pdf = np.zeros((801,761))
minx = 360
maxx = 0
miny = 180
maxy = -180

for i in range(len(sites)):
	for file in sorted(glob.glob(f'/nesi/nobackup/vuw03073/bigboy/all_settlement/{sites[i]}*')):
		
		print(file[-13:])
		traj = nc.Dataset(file)
		
		lon = traj.variables['lon'][:]
		lat = traj.variables['lat'][:]
		x = lon[np.where(lon.mask==False)]
		if np.min(x) < minx:
			minx = np.min(x)
		if np.max(x) > maxx:
			maxx = np.max(x)
		y = lat[np.where(lat.mask==False)]
		if np.min(y) < miny:
			miny = np.min(y)
		if np.max(y) > maxy:
			maxy = np.max(y)
		lon_edges = np.linspace(165, 184, 381)
		lat_edges = np.linspace(-52, -32, 401)
		lon_edges = np.linspace(165, 184, 381*2)
		lat_edges = np.linspace(-52, -32, 401*2)
		H, _, _ = np.histogram2d(y, x, [lat_edges, lon_edges])#, density = True)
		pdfs[i] += H


pdf_south = pdf_CAP + pdf_CAM + pdf_KAI + pdf_GOB + pdf_TIM + pdf_HSB + pdf_BGB + pdf_FIO
pdf_central = pdf_WEST + pdf_FLE + pdf_TAS + pdf_LWR
pdf_north = pdf_OPO + pdf_MAU

m = Basemap(resolution= 'h', llcrnrlon = 165, llcrnrlat = miny-1.5, urcrnrlon = maxx+1.5, urcrnrlat = maxy+1.5, area_thresh = 500)
m.drawmapboundary(fill_color='#DDEEFF')#, zorder = 0)
#m.drawcoastlines(color = '#B1A691')#, zorder = 15)
m.fillcontinents(color='#FDF5E6', lake_color = '#FDF5E6')#, zorder = 20)
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdf[pdf==0]=np.min(pdf[pdf>0])
clr = plt.pcolormesh(xs, ys, pdf+pdf_LWR, norm=colors.LogNorm(), cmap = 'jet')
plt.gca().add_patch(plt.Rectangle((math.floor(lon[0,0]*10)/10, math.floor(lat[0,0]*10)/10), .1, .1, edgecolor = 'black', fill=False))
plt.colorbar(clr)
plt.title('TAS')



#EOF

for file in glob.glob('bigboy_by_year/'): 
	bigboy199 = np.loadtxt('bigboy_out.txt')

summer = np.loadtxt('bigboy_out_summer_3mo.txt')
fall = np.loadtxt('bigboy_out_fall_3mo.txt')
winter = np.loadtxt('bigboy_out_winter_3mo.txt')
spring = np.loadtxt('bigboy_out_spring_3mo.txt')

site_order14 = [8, 12, 5, 6, 7, 4, 13, 11, 10, 9, 3, 2, 1, 0]
summer = summer[:, [30, 80, 81, 157, 173, 187, 199, 235, 247, 249, 271, 316, 432, 442]]
summer = summer[[30, 80, 81, 157, 173, 187, 199, 235, 247, 249, 271, 316, 432, 442], :]
summer = summer[site_order14]
summer = summer[:,site_order14]
site_labs14 = ['FIO', 'BGB', 'HSB', 'TIM', 'LWR', 'WEST', 'FLE', 'TAS', 'OPO', 'GOB', 'KAI', 'CAM', 'MAU', 'CAP']
site_labs = np.asarray(site_labs14)
site_labs = site_labs[site_order14]


fall = fall[:, [30, 80, 81, 157, 173, 187, 199, 235, 247, 249, 271, 316, 432, 442]]
fall = fall[[30, 80, 81, 157, 173, 187, 199, 235, 247, 249, 271, 316, 432, 442], :]
fall = fall[site_order14]
fall = fall[:,site_order14]

winter = winter[:, [30, 80, 81, 157, 173, 187, 199, 235, 247, 249, 271, 316, 432, 442]]
winter = winter[[30, 80, 81, 157, 173, 187, 199, 235, 247, 249, 271, 316, 432, 442], :]
winter = winter[site_order14]
winter = winter[:,site_order14]

spring = spring[:, [30, 80, 81, 157, 173, 187, 199, 235, 247, 249, 271, 316, 432, 442]]
spring = spring[[30, 80, 81, 157, 173, 187, 199, 235, 247, 249, 271, 316, 432, 442], :]
spring = spring[site_order14]
spring = spring[:,site_order14]

seasons = np.array([summer,fall,winter,spring])
seasons_anom = seasons - np.mean(seasons, axis=0)


solver = Eof(seasons)
eofs = solver.eofsAsCorrelation(neofs=4)
eigs = solver.varianceFraction()

fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
axes = axes.flatten()

for i in range(4):
	htmp = sns.heatmap(eofs[i], cmap=plt.cm.RdBu_r, ax=axes[i], cbar = False, xticklabels = site_labs, yticklabels = site_labs)
	axes[i].set_title(f'EOF {i+1}: {eigs[i]*100:.2f}% of variance')
	#axes[i].set_xticks([])
	#axes[i].set_yticks([])
	#if i%2 == 0:
	#axes[i].set_yticks(range(len(site_labs)))
	#axes[i].set_yticklabels(site_labs, rotation = 0)
	#if i >= 2:
	#axes[i].set_xticks(range(len(site_labs)))
	#axes[i].set_xticklabels(site_labs, rotation = 90)
	axes[i].invert_xaxis()

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=colors.Normalize(vmin = -1, vmax=1))
fig.colorbar(cm, cax=cbar_ax)









#!/usr/bin/env python3

import os
import sys
import numpy as np
import netCDF4 as nc
import pandas as pd
import math
import glob
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from mpl_toolkits.basemap import Basemap
from matplotlib import colors
from scipy import stats


from eofs.standard import Eof


#Connectivity matrices

bigboy14 = np.loadtxt('bigboy_all_settlement.txt')
just14 = bigboy14[:, [30, 80, 81, 157, 173, 187, 199, 235, 247, 249, 271, 316, 432, 442]]
just14 = just14[[30, 80, 81, 157, 173, 187, 199, 235, 247, 249, 271, 316, 432, 442], :]
site_order14 = [8, 12, 5, 6, 7, 4, 13, 11, 10, 9, 3, 2, 1, 0]
just14_order = just14[site_order14]
just14_order = just14_order[:,site_order14]
site_labs14 = [r'$\mathregular{FIO_S}$', r'$\mathregular{BGB_S}$', r'$\mathregular{HSB_S}$', r'$\mathregular{TIM_S}$', r'$\mathregular{LWR_S}$', r'$\mathregular{WEST_N}$', r'$\mathregular{FLE_N}$', r'$\mathregular{TAS_N}$', r'$\mathregular{OPO_N}$', r'$\mathregular{GOB_S}$', r'$\mathregular{KAI_S}$', r'$\mathregular{CAM_N}$', r'$\mathregular{MAU_N}$', r'$\mathregular{CAP_N}$']
site_labs = np.asarray(site_labs14)
site_labs = site_labs[site_order14]

just14_order2 = just14_order[fst_site_order]
just14_order2 = just14_order2[:,fst_site_order]
df = pd.DataFrame(data = just14_order, index = fst_site_labs, columns = fst_site_labs)
num=np.sum(bigboy14[30])
df_pct=(df/num)*100
df_log=np.log10(df_pct)
df_log[np.isneginf(df_log)] = -5



ax = sns.heatmap(df_log, cbar_kws={'label': '% Succesful settlement'})#, annot=True, fmt='.0f')
cbar = ax.collections[0].colorbar
cbar.set_ticks([1,0,-1,-2,-3])
cbar.set_ticklabels([10,1,.1,.01,.001])
ax.invert_xaxis()
plt.xlabel("Destination Population")
plt.ylabel("Source Population")
plt.title("Oceanographic Connectivity")
plt.tight_layout()
plt.show()


fst = np.array(([[0,0.014,-0.010,0,0.007,0.033,0.002,0.005,0.019,0.021,0.07,0.027,0.052,0.039],
[0.014,0,0.008,0.019,0.008,0.037,0.022,0.019,0.026,0.03,0.081,0.038,0.071,0.038],
[-0.010,0.008,0,-0.008,-0.001,0.021,-0.007,-0.008,0.005,0.012,0.058,0.013,0.04,0.033],
[0,0.019,-0.008,0,0.011,0.02,0.009,0.006,0,0.01,0.053,0.007,0.038,0.036],
[0.007,0.008,-0.001,0.011,0,0.024,0.004,0.013,0.01,0.014,0.054,0.019,0.042,0.028],
[0.033,0.037,0.021,0.02,0.024,0,0.028,0.029,0.009,0.008,0.036,0,0.03,0.001],
[0.002,0.022,-0.007,0.009,0.004,0.028,0,0.006,0.017,0.019,0.055,0.024,0.043,0.036],
[0.005,0.019,-0.008,0.006,0.013,0.029,0.006,0,0.019,0.031,0.074,0.026,0.063,0.048],
[0.019,0.026,0.005,0,0.01,0.009,0.017,0.019,0,-0.001,0.021,0.006,0.02,0.017],
[0.021,0.03,0.012,0.01,0.014,0.008,0.019,0.031,-0.001,0,0.017,0.008,0.017,0.001],
[0.07,0.081,0.058,0.053,0.054,0.036,0.055,0.074,0.021,0.017,0,0.02,0.011,0.021],
[0.027,0.038,0.013,0.007,0.019,0,0.024,0.026,0.006,0.008,0.02,0,0.013,0.013],
[0.052,0.071,0.04,0.038,0.042,0.03,0.043,0.063,0.02,0.017,0.011,0.013,0,0.028],
[0.039,0.038,0.033,0.036,0.028,0.001,0.036,0.048,0.017,0.001,0.021,0.013,0.028,0]]))

fst[fst<0] = 0

fst_df = pd.DataFrame(data = fst, index = site_labs, columns = site_labs)

cmap = sns.cm.rocket_r
ax = sns.heatmap(fst_df, cbar_kws={'label': 'Fst'}, cmap=cmap)#, annot=True, fmt='.0f')
cbar = ax.collections[0].colorbar
#cbar.set_ticks([1,0,-1,-2,-3])
#cbar.set_ticklabels([10,1,.1,.01,.001])
ax.invert_xaxis()
#plt.xlabel("Destination Population")
#plt.ylabel("Source Population")
#plt.title("Ocean Connectivity")
plt.tight_layout()
plt.title('Genetic connectivity')
plt.show()

fst_site_order = [0,2,3,7,6,4,1,8,9,5,11,13,12,10]
fst_order = fst[fst_site_order]
fst_order = fst_order[:,fst_site_order]
fst_site_labs = [site_labs[i] for i in fst_site_order]
fst_order_df = pd.DataFrame(data = fst_order, index = fst_site_labs, columns = fst_site_labs)

ax = sns.heatmap(fst_order_df, cbar_kws={'label': 'Fst'}, cmap=cmap)#, annot=True, fmt='.0f')
cbar = ax.collections[0].colorbar
#cbar.set_ticks([1,0,-1,-2,-3])
#cbar.set_ticklabels([10,1,.1,.01,.001])
ax.invert_xaxis()
#plt.xlabel("Destination Population")
#plt.ylabel("Source Population")
plt.title("Genetic Connectivity")
plt.tight_layout()
plt.show()

just14_pct = just14_order/num
m, b = np.polyfit(np.log10(just14_pct[just14_pct!=0].flatten()), fst[just14_pct!=0].flatten(), 1)
r = stats.pearsonr(np.log10(just14_pct[just14_pct!=0].flatten()), fst[just14_pct!=0].flatten())

plt.scatter(np.log10(just14_pct.flatten()), fst.flatten(), color = 'black')
plt.plot(np.log10(just14_pct.flatten()), m*np.log10(just14_pct.flatten()) + b, color = 'red')
plt.xlabel('Oceanographic connectivity (log10(% migrants)')
plt.ylabel('Genetic connectivity (Fst)')
plt.title('Correlation coefficient (R) = -.327, p = 2.7e-6')
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

for i in range(len(sites)):
	for file in glob.glob(f'bigboy_pdf/competent/{sites[i]}*'):
		pdfs[i]+=np.loadtxt(file)

pdf = np.zeros((801,761))
minx = 360
maxx = 0
miny = 180
maxy = -180

for i in range(len(sites)):
xs = np.array(())
ys = np.array(())
lon_edges = np.linspace(165, 184, 381)
lat_edges = np.linspace(-52, -32, 401)
lon_edges = np.linspace(165, 184, 381*2)
lat_edges = np.linspace(-52, -32, 401*2)

for file in sorted(glob.glob(f'/nesi/nobackup/vuw03073/bigboy/all_settlement/FIO*')):
	print(file[-13:])
	traj = nc.Dataset(file)
	lon = traj.variables['lon'][:]
	lat = traj.variables['lat'][:]
	x = lon[np.where(lon.mask==False)]
	y = lat[np.where(lat.mask==False)]
	#xs = np.append(xs,x)
	#ys = np.append(ys,y)
	H, _, _ = np.histogram2d(y, x, [lat_edges, lon_edges])#, density = True)
	pdf += H


pdf_OPO = np.loadtxt('bigboy_pdf/competent/OPO_pdf.txt')
pdf_MAU = np.loadtxt('bigboy_pdf/competent/MAU_pdf.txt')
pdf_WEST = np.loadtxt('bigboy_pdf/competent/WEST_pdf.txt')
pdf_FLE = np.loadtxt('bigboy_pdf/competent/FLE_pdf.txt')
pdf_TAS = np.loadtxt('bigboy_pdf/competent/TAS_pdf.txt')
pdf_LWR = np.loadtxt('bigboy_pdf/competent/LWR_pdf.txt')
pdf_CAP = np.loadtxt('bigboy_pdf/competent/CAP_pdf.txt')
pdf_CAM = np.loadtxt('bigboy_pdf/competent/CAM_pdf.txt')
pdf_KAI = np.loadtxt('bigboy_pdf/competent/KAI_pdf.txt')
pdf_GOB = np.loadtxt('bigboy_pdf/competent/GOB_pdf.txt')
pdf_TIM = np.loadtxt('bigboy_pdf/competent/TIM_pdf.txt')
pdf_HSB = np.loadtxt('bigboy_pdf/competent/HSB_pdf.txt')
pdf_BGB = np.loadtxt('bigboy_pdf/competent/BGB_pdf.txt')
pdf_FIO = np.loadtxt('bigboy_pdf/competent/FIO_pdf.txt')





pdf_south = pdf_CAP + pdf_CAM + pdf_KAI + pdf_GOB + pdf_TIM + pdf_HSB + pdf_BGB + pdf_FIO
pdf_central = pdf_WEST + pdf_FLE + pdf_TAS + pdf_LWR
pdf_north = pdf_OPO + pdf_MAU

site_lons=[173.2, 176.0, 172.4, 172.7, 173.1, 176.3, 174.3, 171.9, 173.7, 173.3, 171.3, 168.2, 168.2, 166.8]
site_lats=[-35.5, -37.4, -40.5, -40.6, -41.1, -40.9, -41.7, -41.3, -42.4, -42.9, -44.4, -46.8, -46.9, -45.1]
site_lons=[173.2, 176.0, 172.4, 172.7, 173.1, 171.9, 176.3, 174.3, 173.7, 173.3, 171.3, 168.2, 168.2, 166.8]
site_lats=[-35.5, -37.4, -40.5, -40.6, -41.1, -41.3, -40.9, -41.7, -42.4, -42.9, -44.4, -46.8, -46.9, -45.1]

symbols = ['w^','w^','wo','wo','wo','wo','wo','ws','ws','ws','ws','ws','ws','ws']

lon_edges = np.linspace(165, 184, 381)
lat_edges = np.linspace(-52, -32, 401)
lon_edges = np.linspace(165, 184, 381*2)
lat_edges = np.linspace(-52, -32, 401*2)


cmap = copy.copy(plt.cm.get_cmap("jet"))
cmap.set_bad(cmap(0))


fig, axes = plt.subplots(1,3)
axes = axes.flatten()

m = Basemap(resolution= 'h', llcrnrlon = 165, llcrnrlat = -52, urcrnrlon = 184, urcrnrlat = -32, area_thresh = 500, ax = axes[0])
m.drawmapboundary(fill_color='#DDEEFF', zorder = 0)
m.drawcoastlines(color = '#B1A691', zorder = 15)
parallels = np.arange(-50.,-30,5.)
#m.drawparallels(parallels,labels=[False,True,True,False], linewidth=0)
meridians = np.arange(165.,185,5.)
#m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=0)
axes[0].set_xticks(meridians)
axes[0].set_yticks(parallels)
axes[0].set_xticklabels(['165°E','170°E','175°E', '180°E'])
axes[0].set_yticklabels(['50°S','45°S','40°S','35°S'])
m.fillcontinents(color='#FDF5E6', lake_color = '#FDF5E6', zorder = 20)
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdf = pdf_south / 31050096
#pdf[pdf==0] = 1
clr = axes[0].pcolormesh(xs, ys, pdf, norm=colors.LogNorm(), cmap = cmap)
#for i in range(len(site_lons)):
#	axes[0].add_patch(plt.Rectangle((site_lons[i], site_lats[i]), .1, .1, edgecolor = 'white', fill=False))
for i in range(len(site_lons)):
	axes[0].plot(site_lons[i], site_lats[i], symbols[i], mec = 'k', zorder=30)

#fig.colorbar(clr, ax = axes[0])
axes[0].set_title('a)', loc='left')

m = Basemap(resolution= 'h', llcrnrlon = 165, llcrnrlat = -52, urcrnrlon = 184, urcrnrlat = -32, area_thresh = 500, ax = axes[1])
m.drawmapboundary(fill_color='#DDEEFF', zorder = 0)
m.drawcoastlines(color = '#B1A691', zorder = 15)
parallels = np.arange(-50.,-30,5.)
#m.drawparallels(parallels,labels=[False,True,True,False], linewidth=0)
meridians = np.arange(165.,185,5.)
#m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=0)
axes[1].set_xticks(meridians)
axes[1].set_yticks(parallels)
axes[1].set_xticklabels(['165°E','170°E','175°E', '180°E'])
axes[1].set_yticklabels([])
m.fillcontinents(color='#FDF5E6', lake_color = '#FDF5E6', zorder = 20)
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdf = pdf_central/31050096
#pdf[pdf==0] = 1
clr = axes[1].pcolormesh(xs, ys, pdf, norm=colors.LogNorm(), cmap = cmap)
#for i in range(len(site_lons)):
#	axes[1].add_patch(plt.Rectangle((site_lons[i], site_lats[i]), .1, .1, edgecolor = 'white', fill=False))
for i in range(len(site_lons)):
	axes[1].plot(site_lons[i], site_lats[i], symbols[i], mec = 'k', zorder=30)

#fig.colorbar(clr, ax = axes[1])
axes[1].set_title('b)', loc = 'left')

m = Basemap(resolution= 'h', llcrnrlon = 165, llcrnrlat = -52, urcrnrlon = 184, urcrnrlat = -32, area_thresh = 500, ax = axes[2])
m.drawmapboundary(fill_color='#DDEEFF', zorder = 0)
m.drawcoastlines(color = '#B1A691', zorder = 15)
parallels = np.arange(-50.,-30,5.)
#m.drawparallels(parallels,labels=[False,True,True,False], linewidth=0)
meridians = np.arange(165.,185,5.)
#m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=0)
axes[2].set_xticks(meridians)
axes[2].set_yticks(parallels)
axes[2].set_xticklabels(['165°E','170°E','175°E', '180°E'])
axes[2].set_yticklabels([])
m.fillcontinents(color='#FDF5E6', lake_color = '#FDF5E6', zorder = 20)
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdf = pdf_north/31050096
#pdf[pdf==0] = 1
clr = axes[2].pcolormesh(xs, ys, pdf, norm=colors.LogNorm(), cmap = cmap)
#for i in range(len(site_lons)):
#	axes[2].add_patch(plt.Rectangle((site_lons[i], site_lats[i]), .1, .1, edgecolor = 'white', fill=False))
for i in range(len(site_lons)):
	axes[2].plot(site_lons[i], site_lats[i], symbols[i], mec = 'k', zorder=30)

#fig.colorbar(clr, ax = axes[2])
axes[2].set_title('c)', loc='left')

#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#divider = make_axes_locatable(axes.ravel().tolist())
#cax = divider.append_axes("right", "5%", pad="3%")
#plt.colorbar(clr, cax=cax)
fig.tight_layout()
plt.subplots_adjust(top=0.966,
bottom=0.034,
left=0.047,
right=0.988,
hspace=0.2,
wspace=0.098)

cbar = fig.colorbar(clr, ax=axes.ravel().tolist(), shrink = .75)
cbar.set_label(label = 'Probability density function of particle trajectories (log scale)', fontsize = 8)
cbar.minorticks_off()
plt.show()




fig, axes = plt.subplots(3,5)
axes = axes.flatten()

for i in range(len(pdfs)):
	m = Basemap(resolution= 'h', llcrnrlon = 165, llcrnrlat = -52, urcrnrlon = 184, urcrnrlat = -32, area_thresh = 500, ax = axes[i])
	m.drawmapboundary(fill_color='#DDEEFF')#, zorder = 0)
	m.fillcontinents(color='#FDF5E6', lake_color = '#FDF5E6')#, zorder = 20)
	lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)
	xs, ys = m(lon_bins_2d, lat_bins_2d)
	pdf = pdfs[i]
	pdf[pdf==0] = 1
	clr = axes[i].pcolormesh(xs, ys, pdf, norm=colors.LogNorm(), cmap = 'jet')
	for i in range(len(site_lons)):
		axes[i].add_patch(plt.Rectangle((site_lons[i], site_lats[i]), .1, .1, edgecolor = 'black', fill=False))
	#axes[1].colorbar(clr)
	#axes[1].set_title('Southern populations')

plt.show()



# LWR variation 


lwr_summer = np.zeros((801,761))
for file in glob.glob('bigboy_pdf/oneeighty/LWR/*01_pdf.txt'):
	lwr_summer += np.loadtxt(file)

for file in glob.glob('bigboy_pdf/oneeighty/LWR/*02_pdf.txt'):
	lwr_summer += np.loadtxt(file)

for file in glob.glob('bigboy_pdf/oneeighty/LWR/*03_pdf.txt'):
	lwr_summer += np.loadtxt(file)


lwr_fall = np.zeros((801,761))
for file in glob.glob('bigboy_pdf/oneeighty/LWR/*04_pdf.txt'):
	lwr_fall += np.loadtxt(file)

for file in glob.glob('bigboy_pdf/oneeighty/LWR/*05_pdf.txt'):
	lwr_fall += np.loadtxt(file)

for file in glob.glob('bigboy_pdf/oneeighty/LWR/*06_pdf.txt'):
	lwr_fall += np.loadtxt(file)


lwr_winter = np.zeros((801,761))
for file in glob.glob('bigboy_pdf/oneeighty/LWR/*07_pdf.txt'):
	lwr_winter += np.loadtxt(file)

for file in glob.glob('bigboy_pdf/oneeighty/LWR/*08_pdf.txt'):
	lwr_winter += np.loadtxt(file)

for file in glob.glob('bigboy_pdf/oneeighty/LWR/*09_pdf.txt'):
	lwr_winter += np.loadtxt(file)


lwr_spring = np.zeros((801,761))
for file in glob.glob('bigboy_pdf/oneeighty/LWR/*10_pdf.txt'):
	lwr_spring += np.loadtxt(file)

for file in glob.glob('bigboy_pdf/oneeighty/LWR/*11_pdf.txt'):
	lwr_spring += np.loadtxt(file)

for file in glob.glob('bigboy_pdf/oneeighty/LWR/*12_pdf.txt'):
	lwr_spring += np.loadtxt(file)



fig, axes = plt.subplots(2,2)
axes = axes.flatten()
m = Basemap(resolution= 'h', llcrnrlon = 167, llcrnrlat = -45, urcrnrlon = 177, urcrnrlat = -35, area_thresh = 500, ax = axes[0])
m.drawmapboundary(fill_color='#DDEEFF', zorder = 0)
m.drawcoastlines(color = '#B1A691', zorder = 15)
parallels = np.arange(-44.,-34,2.)
#m.drawparallels(parallels,labels=[False,True,True,False], linewidth=0)
meridians = np.arange(168.,178,2.)
#m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=0)
axes[0].set_xticks(meridians)
axes[0].set_yticks(parallels)
axes[0].set_xticklabels([])
axes[0].set_yticklabels(['44°S','42°S','40°S','38°S', '36°S'])
m.fillcontinents(color='#FDF5E6', lake_color = '#FDF5E6', zorder = 20)
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdf = lwr_summer
pdf[pdf==0] = 1
clr = axes[0].pcolormesh(xs, ys, pdf, norm=colors.LogNorm(), cmap = 'jet')
#for i in range(len(site_lons)):
#	axes[0].add_patch(plt.Rectangle((site_lons[i], site_lats[i]), .1, .1, edgecolor = 'white', fill=False))
for i in range(len(site_lons)):
	axes[0].plot(site_lons[i], site_lats[i], symbols[i], mec = 'k', zorder=30)

#axes[0].colorbar(clr)
axes[0].set_title('a)', loc='left')

m = Basemap(resolution= 'h', llcrnrlon = 167, llcrnrlat = -45, urcrnrlon = 177, urcrnrlat = -35, area_thresh = 500, ax = axes[1])
m.drawmapboundary(fill_color='#DDEEFF', zorder = 0)
m.drawcoastlines(color = '#B1A691', zorder = 15)
parallels = np.arange(-44.,-34,2.)
#m.drawparallels(parallels,labels=[False,True,True,False], linewidth=0)
meridians = np.arange(168.,178,2.)
#m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=0)
axes[1].set_xticks(meridians)
axes[1].set_yticks(parallels)
axes[1].set_xticklabels([])
axes[1].set_yticklabels([])
m.fillcontinents(color='#FDF5E6', lake_color = '#FDF5E6', zorder = 20)
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdf = lwr_fall
pdf[pdf==0] = 1
clr = axes[1].pcolormesh(xs, ys, pdf, norm=colors.LogNorm(), cmap = 'jet')
#for i in range(len(site_lons)):
#	axes[1].add_patch(plt.Rectangle((site_lons[i], site_lats[i]), .1, .1, edgecolor = 'white', fill=False))
for i in range(len(site_lons)):
	axes[1].plot(site_lons[i], site_lats[i], symbols[i], mec = 'k', zorder=30)

#axes[1].colorbar(clr)
axes[1].set_title('b)', loc = 'left')

m = Basemap(resolution= 'h', llcrnrlon = 167, llcrnrlat = -45, urcrnrlon = 177, urcrnrlat = -35, area_thresh = 500, ax = axes[2])
m.drawmapboundary(fill_color='#DDEEFF', zorder = 0)
m.drawcoastlines(color = '#B1A691', zorder = 15)
parallels = np.arange(-44.,-34,2.)
#m.drawparallels(parallels,labels=[False,True,True,False], linewidth=0)
meridians = np.arange(168.,178,2.)
#m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=0)
axes[2].set_xticks(meridians)
axes[2].set_yticks(parallels)
axes[2].set_xticklabels(['168°E','170°E','172°E','174°E','176°E'])
axes[2].set_yticklabels(['44°S','42°S','40°S','38°S', '36°S'])
m.fillcontinents(color='#FDF5E6', lake_color = '#FDF5E6', zorder = 20)
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdf = lwr_winter
pdf[pdf==0] = 1
clr = axes[2].pcolormesh(xs, ys, pdf, norm=colors.LogNorm(), cmap = 'jet')
#for i in range(len(site_lons)):
#	axes[2].add_patch(plt.Rectangle((site_lons[i], site_lats[i]), .1, .1, edgecolor = 'white', fill=False))
for i in range(len(site_lons)):
	axes[2].plot(site_lons[i], site_lats[i], symbols[i], mec = 'k', zorder=30)

#axes[2].colorbar(clr)
axes[2].set_title('c)', loc='left')

m = Basemap(resolution= 'h', llcrnrlon = 167, llcrnrlat = -45, urcrnrlon = 177, urcrnrlat = -35, area_thresh = 500, ax = axes[3])
m.drawmapboundary(fill_color='#DDEEFF', zorder = 0)
m.drawcoastlines(color = '#B1A691', zorder = 15)
parallels = np.arange(-44.,-34,2.)
#m.drawparallels(parallels,labels=[False,True,True,False], linewidth=0)
meridians = np.arange(168.,178,2.)
#m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=0)
axes[3].set_xticks(meridians)
axes[3].set_yticks(parallels)
axes[3].set_xticklabels(['168°E','170°E','172°E','174°E','176°E'])
axes[3].set_yticklabels([])
m.fillcontinents(color='#FDF5E6', lake_color = '#FDF5E6', zorder = 20)
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdf = lwr_spring
pdf[pdf==0] = 1
clr = axes[3].pcolormesh(xs, ys, pdf, norm=colors.LogNorm(), cmap = 'jet')
#for i in range(len(site_lons)):
#	axes[3].add_patch(plt.Rectangle((site_lons[i], site_lats[i]), .1, .1, edgecolor = 'white', fill=False))
for i in range(len(site_lons)):
	axes[3].plot(site_lons[i], site_lats[i], symbols[i], mec = 'k', zorder=30)

#axes[3].colorbar(clr)
axes[3].set_title('d)', loc='left')


plt.show()






#KAI_GOB


kai_gob_summer = np.zeros((801,761))
for file in glob.glob('bigboy_pdf/oneeighty/KAI_GOB/*01_pdf.txt'):
	kai_gob_summer += np.loadtxt(file)

for file in glob.glob('bigboy_pdf/oneeighty/KAI_GOB/*02_pdf.txt'):
	kai_gob_summer += np.loadtxt(file)

for file in glob.glob('bigboy_pdf/oneeighty/KAI_GOB/*03_pdf.txt'):
	kai_gob_summer += np.loadtxt(file)


kai_gob_fall = np.zeros((801,761))
for file in glob.glob('bigboy_pdf/oneeighty/KAI_GOB/*04_pdf.txt'):
	kai_gob_fall += np.loadtxt(file)

for file in glob.glob('bigboy_pdf/oneeighty/KAI_GOB/*05_pdf.txt'):
	kai_gob_fall += np.loadtxt(file)

for file in glob.glob('bigboy_pdf/oneeighty/KAI_GOB/*06_pdf.txt'):
	kai_gob_fall += np.loadtxt(file)


kai_gob_winter = np.zeros((801,761))
for file in glob.glob('bigboy_pdf/oneeighty/KAI_GOB/*07_pdf.txt'):
	kai_gob_winter += np.loadtxt(file)

for file in glob.glob('bigboy_pdf/oneeighty/KAI_GOB/*08_pdf.txt'):
	kai_gob_winter += np.loadtxt(file)

for file in glob.glob('bigboy_pdf/oneeighty/KAI_GOB/*09_pdf.txt'):
	kai_gob_winter += np.loadtxt(file)


kai_gob_spring = np.zeros((801,761))
for file in glob.glob('bigboy_pdf/oneeighty/KAI_GOB/*10_pdf.txt'):
	kai_gob_spring += np.loadtxt(file)

for file in glob.glob('bigboy_pdf/oneeighty/KAI_GOB/*11_pdf.txt'):
	kai_gob_spring += np.loadtxt(file)

for file in glob.glob('bigboy_pdf/oneeighty/KAI_GOB/*12_pdf.txt'):
	kai_gob_spring += np.loadtxt(file)



fig, axes = plt.subplots(2,2)
axes = axes.flatten()
m = Basemap(resolution= 'h', llcrnrlon = 170, llcrnrlat = -45, urcrnrlon = 184, urcrnrlat = -35, area_thresh = 500, ax = axes[0])
m.drawmapboundary(fill_color='#DDEEFF', zorder = 0)
m.drawcoastlines(color = '#B1A691', zorder = 15)
parallels = np.arange(-44.,-34,2.)
#m.drawparallels(parallels,labels=[False,True,True,False], linewidth=0)
meridians = np.arange(170.,184,2.)
#m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=0)
axes[0].set_xticks(meridians)
axes[0].set_yticks(parallels)
axes[0].set_xticklabels([])
axes[0].set_yticklabels(['44°S','42°S','40°S','38°S', '36°S'])
m.fillcontinents(color='#FDF5E6', lake_color = '#FDF5E6', zorder = 20)
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdf = kai_gob_summer
pdf[pdf==0] = 1
clr = axes[0].pcolormesh(xs, ys, pdf, norm=colors.LogNorm(), cmap = 'jet')
#for i in range(len(site_lons)):
#	axes[0].add_patch(plt.Rectangle((site_lons[i], site_lats[i]), .1, .1, edgecolor = 'white', fill=False))
for i in range(len(site_lons)):
	axes[0].plot(site_lons[i], site_lats[i], symbols[i], mec = 'k', zorder=30)

#axes[0].colorbar(clr)
axes[0].set_title('a)', loc='left')

m = Basemap(resolution= 'h', llcrnrlon = 170, llcrnrlat = -45, urcrnrlon = 184, urcrnrlat = -35, area_thresh = 500, ax = axes[1])
m.drawmapboundary(fill_color='#DDEEFF', zorder = 0)
m.drawcoastlines(color = '#B1A691', zorder = 15)
parallels = np.arange(-44.,-34,2.)
#m.drawparallels(parallels,labels=[False,True,True,False], linewidth=0)
meridians = np.arange(170.,184,2.)
#m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=0)
axes[1].set_xticks(meridians)
axes[1].set_yticks(parallels)
axes[1].set_xticklabels([])
axes[1].set_yticklabels([])
m.fillcontinents(color='#FDF5E6', lake_color = '#FDF5E6', zorder = 20)
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdf = kai_gob_fall
pdf[pdf==0] = 1
clr = axes[1].pcolormesh(xs, ys, pdf, norm=colors.LogNorm(), cmap = 'jet')
#for i in range(len(site_lons)):
#	axes[1].add_patch(plt.Rectangle((site_lons[i], site_lats[i]), .1, .1, edgecolor = 'white', fill=False))
for i in range(len(site_lons)):
	axes[1].plot(site_lons[i], site_lats[i], symbols[i], mec = 'k', zorder=30)

#axes[1].colorbar(clr)
axes[1].set_title('b)', loc = 'left')

m = Basemap(resolution= 'h', llcrnrlon = 170, llcrnrlat = -45, urcrnrlon = 184, urcrnrlat = -35, area_thresh = 500, ax = axes[2])
m.drawmapboundary(fill_color='#DDEEFF', zorder = 0)
m.drawcoastlines(color = '#B1A691', zorder = 15)
parallels = np.arange(-44.,-34,2.)
#m.drawparallels(parallels,labels=[False,True,True,False], linewidth=0)
meridians = np.arange(170.,184.1,2.)
#m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=0)
axes[2].set_xticks(meridians)
axes[2].set_yticks(parallels)
axes[2].set_xticklabels(['170°E','172°E','174°E','176°E','178°E','180°E','178°W', '176°W'])
axes[2].set_yticklabels(['44°S','42°S','40°S','38°S', '36°S'])
m.fillcontinents(color='#FDF5E6', lake_color = '#FDF5E6', zorder = 20)
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdf = kai_gob_winter
pdf[pdf==0] = 1
clr = axes[2].pcolormesh(xs, ys, pdf, norm=colors.LogNorm(), cmap = 'jet')
#for i in range(len(site_lons)):
#	axes[2].add_patch(plt.Rectangle((site_lons[i], site_lats[i]), .1, .1, edgecolor = 'white', fill=False))
for i in range(len(site_lons)):
	axes[2].plot(site_lons[i], site_lats[i], symbols[i], mec = 'k', zorder=30)

#axes[2].colorbar(clr)
axes[2].set_title('c)', loc='left')

m = Basemap(resolution= 'h', llcrnrlon = 170, llcrnrlat = -45, urcrnrlon = 184, urcrnrlat = -35, area_thresh = 500, ax = axes[3])
m.drawmapboundary(fill_color='#DDEEFF', zorder = 0)
m.drawcoastlines(color = '#B1A691', zorder = 15)
parallels = np.arange(-44.,-34,2.)
#m.drawparallels(parallels,labels=[False,True,True,False], linewidth=0)
meridians = np.arange(170.,184.1,2.)
#m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=0)
axes[3].set_xticks(meridians)
axes[3].set_yticks(parallels)
axes[3].set_xticklabels(['170°E','172°E','174°E','176°E','178°E','180°E','178°W', '176°W'])
axes[3].set_yticklabels([])
m.fillcontinents(color='#FDF5E6', lake_color = '#FDF5E6', zorder = 20)
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdf = kai_gob_spring
pdf[pdf==0] = 1
clr = axes[3].pcolormesh(xs, ys, pdf, norm=colors.LogNorm(), cmap = 'jet')
#for i in range(len(site_lons)):
#	axes[3].add_patch(plt.Rectangle((site_lons[i], site_lats[i]), .1, .1, edgecolor = 'white', fill=False))
for i in range(len(site_lons)):
	axes[3].plot(site_lons[i], site_lats[i], symbols[i], mec = 'k', zorder=30)

#axes[3].colorbar(clr)
axes[3].set_title('d)', loc='left')


plt.show()














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








fig, axes = plt.subplots(1,3)
axes = axes.flatten()

m = Basemap(resolution= 'h', llcrnrlon = 165, llcrnrlat = -52, urcrnrlon = 184, urcrnrlat = -32, area_thresh = 500, ax = axes[0])
m.drawmapboundary(fill_color='#DDEEFF', zorder = 0)
m.drawcoastlines(color = '#B1A691', zorder = 15)
parallels = np.arange(-50.,-30,5.)
#m.drawparallels(parallels,labels=[False,True,True,False], linewidth=0)
meridians = np.arange(165.,185,5.)
#m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=0)
axes[0].set_xticks(meridians)
axes[0].set_yticks(parallels)
axes[0].set_xticklabels(['165°E','170°E','175°E', '180°E'])
axes[0].set_yticklabels(['50°S','45°S','40°S','35°S'])
m.fillcontinents(color='#FDF5E6', lake_color = '#FDF5E6', zorder = 20)
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdf = pdf_WEST
pdf[pdf==0] = 1
clr = axes[0].pcolormesh(xs, ys, pdf, norm=colors.LogNorm(), cmap = 'jet')
#for i in range(len(site_lons)):
#	axes[0].add_patch(plt.Rectangle((site_lons[i], site_lats[i]), .1, .1, edgecolor = 'white', fill=False))
for i in range(len(site_lons)):
	axes[0].plot(site_lons[i], site_lats[i], symbols[i], mec = 'k', zorder=30)

#fig.colorbar(clr, ax = axes[0])
axes[0].set_title('a)', loc='left')

m = Basemap(resolution= 'h', llcrnrlon = 165, llcrnrlat = -52, urcrnrlon = 184, urcrnrlat = -32, area_thresh = 500, ax = axes[1])
m.drawmapboundary(fill_color='#DDEEFF', zorder = 0)
m.drawcoastlines(color = '#B1A691', zorder = 15)
parallels = np.arange(-50.,-30,5.)
#m.drawparallels(parallels,labels=[False,True,True,False], linewidth=0)
meridians = np.arange(165.,185,5.)
#m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=0)
axes[1].set_xticks(meridians)
axes[1].set_yticks(parallels)
axes[1].set_xticklabels(['165°E','170°E','175°E', '180°E'])
axes[1].set_yticklabels([])
m.fillcontinents(color='#FDF5E6', lake_color = '#FDF5E6', zorder = 20)
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdf = pdf_CAM
pdf[pdf==0] = 1
clr = axes[1].pcolormesh(xs, ys, pdf, norm=colors.LogNorm(), cmap = 'jet')
#for i in range(len(site_lons)):
#	axes[1].add_patch(plt.Rectangle((site_lons[i], site_lats[i]), .1, .1, edgecolor = 'white', fill=False))
for i in range(len(site_lons)):
	axes[1].plot(site_lons[i], site_lats[i], symbols[i], mec = 'k', zorder=30)

#fig.colorbar(clr, ax = axes[1])
axes[1].set_title('b)', loc = 'left')

m = Basemap(resolution= 'h', llcrnrlon = 165, llcrnrlat = -52, urcrnrlon = 184, urcrnrlat = -32, area_thresh = 500, ax = axes[2])
m.drawmapboundary(fill_color='#DDEEFF', zorder = 0)
m.drawcoastlines(color = '#B1A691', zorder = 15)
parallels = np.arange(-50.,-30,5.)
#m.drawparallels(parallels,labels=[False,True,True,False], linewidth=0)
meridians = np.arange(165.,185,5.)
#m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=0)
axes[2].set_xticks(meridians)
axes[2].set_yticks(parallels)
axes[2].set_xticklabels(['165°E','170°E','175°E', '180°E'])
axes[2].set_yticklabels([])
m.fillcontinents(color='#FDF5E6', lake_color = '#FDF5E6', zorder = 20)
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)
xs, ys = m(lon_bins_2d, lat_bins_2d)
pdf = pdf_CAP
pdf[pdf==0] = 1
clr = axes[2].pcolormesh(xs, ys, pdf, norm=colors.LogNorm(), cmap = 'jet')
#for i in range(len(site_lons)):
#	axes[2].add_patch(plt.Rectangle((site_lons[i], site_lats[i]), .1, .1, edgecolor = 'white', fill=False))
for i in range(len(site_lons)):
	axes[2].plot(site_lons[i], site_lats[i], symbols[i], mec = 'k', zorder=30)

#fig.colorbar(clr, ax = axes[2])
axes[2].set_title('c)', loc='left')
plt.show()


#PLot sourciness
settlement_success = [np.sum(i[0:-1])/np.sum(i) for i in mat1 if np.sum(i) > 30000]
settlement_success = [settlement_success[i] for i in site_order14]

downstream_pops = [len(np.where(i>0)[0]) for i in bigboy if np.sum(i) > 30000]
downstream_pops = [downstream_pops[i] for i in site_order14]

fig, ax = plt.subplots(1,1)

m = Basemap(resolution= 'h', llcrnrlon = 165, llcrnrlat = -52, urcrnrlon = 184, urcrnrlat = -32, area_thresh = 500, ax = ax)
m.drawmapboundary(fill_color='#DDEEFF', zorder = 0)
m.drawcoastlines(color = '#B1A691', zorder = 15)
parallels = np.arange(-50.,-30,5.)
#m.drawparallels(parallels,labels=[False,True,True,False], linewidth=0)
meridians = np.arange(165.,185,5.)
#m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=0)
ax.set_xticks(meridians)
ax.set_yticks(parallels)
ax.set_xticklabels(['165°E','170°E','175°E', '180°E'])
ax.set_yticklabels(['50°S','45°S','40°S','35°S'])
m.fillcontinents(color='#FDF5E6', lake_color = '#FDF5E6', zorder = 20)
#lon_bins_2d, lat_bins_2d = np.meshgrid(lon_edges, lat_edges)
#xs, ys = m(lon_bins_2d, lat_bins_2d)
#pdf = pdf_south / 31050096
for i in range(len(site_lons)):
	weight = (settlement_success[i]/max(settlement_success))
	#colorhex = "#"+str(colorn)*6
	ax.plot(site_lons[i], site_lats[i], 'o', markersize = 20*(downstream_pops[i]/max(downstream_pops)), c = 'green', alpha = weight, zorder=30)

#fig.colorbar(clr, ax = ax)
ax.set_title('a)', loc='left')
plt.show()


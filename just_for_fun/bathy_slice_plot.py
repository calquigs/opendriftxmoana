#!/usr/bin/env python

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

bb = nc.Dataset('/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201512.nc')
z_rho = bb.variables['z_rho'][:]
lon = bb.variables['lon_rho'][:]
lat = bb.variables['lat_rho'][:]

#targetlon = 168.181034
#targetlat = -46.909673
#closestlon = [abs(lon[0,i]-targetlon) for i in range(len(lon[0]))]
#closestlon = closestlon.index(min(closestlon))
#closestlat = [abs(lat[i,0]-targetlat) for i in range(len(lat))]
#closestlat = closestlat.index(min(closestlat))

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 8}

matplotlib.rc('font', **font)


fig, axes = plt.subplots(5,3)

#OPO
x = lon[377, 200:205]
a = z_rho[0,0,377,200:204].tolist()
s = np.zeros((50,5))
s[:,:-1] = z_rho[0,:,377,200:204]
a.append(0)

axes[0,0].plot(x, a, '-ok')
axes[0,0].fill_between(x, a, np.min(a), color = 'black')

for sigma in s:
	axes[0,0].plot(x, sigma, '-', color='0.75', linewidth=.5)

axes[0,0].set_title('Bathymetry at latitude -35.55 (Opononi)')
axes[0,0].set_xlabel('Longitude')
axes[0,0].set_ylabel('Depth (m)')

#MAU
x = lon[338, 247:252]
a = z_rho[0,0,338,248:252].tolist()
s = np.zeros((50,5))
s[:,1:] = z_rho[0,:,338,248:252]
a.insert(0, 0)

axes[0,1].plot(x, a, '-ok')
axes[0,1].fill_between(x, a, np.min(a), color = 'black')

for sigma in s:
	axes[0,1].plot(x, sigma, '-', color='0.75', linewidth=.5)

axes[0,1].set_title('Bathymetry at latitude -37.45 (Maunganui)')
axes[0,1].set_xlabel('Longitude')
axes[0,1].set_ylabel('Depth (m)')

#CAP
x = lon[265, 251:256]
a = z_rho[0,0,265,252:256].tolist()
s = np.zeros((50,5))
s[:,1:] = z_rho[0,:,265,252:256]
a.insert(0,0)

axes[0,2].plot(x, a, '-ok')
axes[0,2].fill_between(x, a, np.min(a), color = 'black')

for sigma in s:
	axes[0,2].plot(x, sigma, '-', color='0.75', linewidth=.5)

axes[0,2].set_title('Bathymetry at latitude -40.91 (Castlepoint)')
axes[0,2].set_xlabel('Longitude')
axes[0,2].set_ylabel('Depth (m)')

#WEST
x = lat[271:276, 189]
a = z_rho[0,0,272:276, 189].tolist()
s = np.zeros((50,5))
s[:,1:] = z_rho[0,:,272:276, 189]
a.insert(0,0)

axes[1,0].plot(x, a, '-ok')
axes[1,0].fill_between(x, a, np.min(a), color = 'black')

for sigma in s:
	axes[1,0].plot(x, sigma, '-', color='0.75', linewidth=.5)

axes[1,0].set_title('Bathymetry at Longitude 172.46 (Westhaven)')
axes[1,0].set_xlabel('Latitude')
axes[1,0].set_ylabel('Depth (m)')
axes[1,0].invert_xaxis()

#FLE
x = lat[270:275, 195]
a = z_rho[0,0,270:274, 195].tolist()
s = np.zeros((50,5))
s[:,:-1] = z_rho[0,:,270:274, 195]
a.append(0)

axes[1,1].plot(x, a, '-ok')
axes[1,1].fill_between(x, a, np.min(a), color = 'black')

for sigma in s:
	axes[1,1].plot(x, sigma, '-', color='0.75', linewidth=.5)

axes[1,1].set_title('Bathymetry at Longitude 172.79 (Fletchers Beach)')
axes[1,1].set_xlabel('Latitude')
axes[1,1].set_ylabel('Depth (m)')
axes[1,1].invert_xaxis()

#TAS
x = lat[257:262, 200]
a = z_rho[0,0,258:262, 200].tolist()
s = np.zeros((50,5))
s[:,1:] = z_rho[0,:,258:262, 200]
a.insert(0,0)

axes[1,2].plot(x, a, '-ok')
axes[1,2].fill_between(x, a, np.min(a), color = 'black')

for sigma in s:
	axes[1,2].plot(x, sigma, '-', color='0.75', linewidth=.5)

axes[1,2].set_title('Bathymetry at Longitude 173.12 (Tasman Bay)')
axes[1,2].set_xlabel('Latitude')
axes[1,2].set_ylabel('Depth (m)')

#CAM
x = lon[247, 219:224]
a = z_rho[0,0,247, 220:224].tolist()
s = np.zeros((50,5))
s[:,1:] = z_rho[0,:,247, 220:224]
a.insert(0,0)

axes[2,0].plot(x, a, '-ok')
axes[2,0].fill_between(x, a, np.min(a), color = 'black')

for sigma in s:
	axes[2,0].plot(x, sigma, '-', color='0.75', linewidth=.5)

axes[2,0].set_title('Bathymetry at Latitude -41.73 (Cape Campbell)')
axes[2,0].set_xlabel('Longitude')
axes[2,0].set_ylabel('Depth (m)')

#LWR
x = lon[255, 178:183]
a = z_rho[0,0,255, 178:182].tolist()
s = np.zeros((50,5))
s[:,:-1] = z_rho[0,:,255, 178:182]
a.append(0)

axes[2,1].plot(x, a, '-ok')
axes[2,1].fill_between(x, a, np.min(a), color = 'black')

for sigma in s:
	axes[2,1].plot(x, sigma, '-', color='0.75', linewidth=.5)

axes[2,1].set_title('Bathymetry at Latitude -41.37 (Little Wanganui River)')
axes[2,1].set_xlabel('Longitude')
axes[2,1].set_ylabel('Depth (m)')


#KAI
x = lon[231, 207:212]
a = z_rho[0,0,231, 208:212].tolist()
s = np.zeros((50,5))
s[:,1:] = z_rho[0,:,231, 208:212]
a.insert(0,0)

axes[2,2].plot(x, a, '-ok')
axes[2,2].fill_between(x, a, np.min(a), color = 'black')

for sigma in s:
	axes[2,2].plot(x, sigma, '-', color='0.75', linewidth=.5)

axes[2,2].set_title('Bathymetry at Latitude -42.44 (Kaikoura)')
axes[2,2].set_xlabel('Longitude')
axes[2,2].set_ylabel('Depth (m)')

#GOB
x = lon[222, 203:208]
a = z_rho[0,0,222, 204:208].tolist()
s = np.zeros((50,5))
s[:,1:] = z_rho[0,:,222, 204:208]
a.insert(0,0)

axes[3,0].plot(x, a, '-ok')
axes[3,0].fill_between(x, a, np.min(a), color = 'black')

for sigma in s:
	axes[3,0].plot(x, sigma, '-', color='0.75', linewidth=.5)

axes[3,0].set_title('Bathymetry at Latitude -42.86 (Gore Bay)')
axes[3,0].set_xlabel('Longitude')
axes[3,0].set_ylabel('Depth (m)')

#TIM
x = lon[187, 169:174]
a = z_rho[0,0,187, 170:174].tolist()
s = np.zeros((50,5))
s[:,1:] = z_rho[0,:,187, 170:174]
a.insert(0,0)

axes[3,1].plot(x, a, '-ok')
axes[3,1].fill_between(x, a, np.min(a), color = 'black')

for sigma in s:
	axes[3,1].plot(x, sigma, '-', color='0.75', linewidth=.5)

axes[3,1].set_title('Bathymetry at Latitude -44.41 (Timaru)')
axes[3,1].set_xlabel('Longitude')
axes[3,1].set_ylabel('Depth (m)')

#FIO
x = lon[168, 93:98]
a = z_rho[0,0,168, 93:97].tolist()
s = np.zeros((50,5))
s[:,:-1] = z_rho[0,:,168, 93:97]
a.append(0)

axes[3,2].plot(x, a, '-ok')
axes[3,2].fill_between(x, a, np.min(a), color = 'black')

for sigma in s:
	axes[3,2].plot(x, sigma, '-', color='0.75', linewidth=.5)

axes[3,2].set_title('Bathymetry at Latitude -45.23 (Fiordland)')
axes[3,2].set_xlabel('Longitude')
axes[3,2].set_ylabel('Depth (m)')

#HSB
x = lon[129, 117:122]
a = z_rho[0,0,129, 118:122].tolist()
s = np.zeros((50,5))
s[:,1:] = z_rho[0,:,129, 118:122]
a.insert(0,0)

axes[4,0].plot(x, a, '-ok')
axes[4,0].fill_between(x, a, np.min(a), color = 'black')

for sigma in s:
	axes[4,0].plot(x, sigma, '-', color='0.75', linewidth=.5)

axes[4,0].set_title('Bathymetry at Latitude -46.86 (Horseshoe Bay)')
axes[4,0].set_xlabel('Longitude')
axes[4,0].set_ylabel('Depth (m)')

#BGB
x = lon[247, 219:224]
a = z_rho[0,0,247, 220:224].tolist()
s = np.zeros((50,5))
s[:,1:] = z_rho[0,:,247, 220:224]
a.insert(0,0)

axes[4,1].plot(x, a, '-ok')
axes[4,1].fill_between(x, a, np.min(a), color = 'black')

for sigma in s:
	axes[4,1].plot(x, sigma, '-', color='0.75', linewidth=.5)

axes[4,1].set_title('Bathymetry at Latitude -41.73 (Big Glory Bay)')
axes[4,1].set_xlabel('Latitude')
axes[4,1].set_ylabel('Depth (m)')


plt.tight_layout()

plt.show()


#!/usr/bin/env python3


import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

path = 'Desktop/opendrift/data/nz5km_his_201707.nc'
path = '/nesi/nobackup/mocean02574/NZB_3/nz5km_his_201707.nc'


bb = nc.Dataset(path)
lons = bb.variables['lon_rho'][0,:]
lats = bb.variables['lat_rho'][:,0]
mask = bb.variables['mask_rho'][:]


m = Basemap(resolution= 'h', llcrnrlon = 154, llcrnrlat = -52, urcrnrlon = 186, urcrnrlat = -29, lon_0 = (161+186)/2, lat_0 = (-50+-28)/2, projection='tmerc', area_thresh = 500)
m.drawcoastlines(color = '#B1A691', zorder = 15)
m.drawmapboundary(fill_color='#DDEEFF', zorder = 0)
m.fillcontinents(color='#FDF5E6', lake_color ='#FDF5E6', zorder = 20)
m.drawparallels(np.arange(-60,-25.,5.), labels=[True,True,False,False], color = '#999999', dashes = [1,1], zorder = 9)
m.drawmeridians(np.arange(150.,195.,5.), labels=[False,False,True,True], color = '#999999', dashes = [1,1], zorder = 10)

for i in range(len(lons[::10])):
    for j in range(len(lats[::10])):
        if mask[::10,::10][j,i]:
            xpt, ypt = m(lons[::10][i], lats[::10][j])
            m.plot(xpt, ypt, 'b.', markersize = 1),# latlon = True)

plt.show()


import geopy.distance

b_crnr = (lats[0], lons[0])
b_crnr_one_over = (lats[0], lons[1])
b_crnr_one_up = (lats[1], lons[0])

u_crnr = (lats[-1], lons[0])
u_crnr_one_over = (lats[-1], lons[1])
u_crnr_one_down = (lats[-2], lons[0])


b_lon_step = geopy.distance.distance(b_crnr, b_crnr_one_over)
b_lat_step = geopy.distance.distance(b_crnr, b_crnr_one_up)
u_lon_step = geopy.distance.distance(u_crnr, u_crnr_one_over)
u_lat_step = geopy.distance.distance(u_crnr, u_crnr_one_down)


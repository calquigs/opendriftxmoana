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
        sfpoints.append([Point(float(elems[0]), float(elems[1])), Point(float(elems[2]), float(elems[3])), elems[4]])
    return sfpoints

allpoints = []
for file in glob.glob('OPO/*'):
    allpoints.append(customout_to_startfinal_points(file))


########################################
#read in settlement bins and create grid
########################################
shape_filename = "all_reef_bins/all_reef_bins.shp"
shp = shapefile.Reader(shape_filename)
bins = shp.shapes()
records = shp.records()
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

########
#plot grid as bins
########
for col in range(grid.nlon):
    for row in range(grid.nlat):
        if grid.bin_idx[col, row] > 0:
            cell_minlon = grid.min_lon+(col)*grid.cell_size
            cell_maxlon = grid.min_lon+(col+1)*grid.cell_size
            cell_maxlat = grid.min_lat+(row+1)*grid.cell_size
            cell_minlat = grid.min_lat+(row)*grid.cell_size
            xs = [cell_minlon, cell_maxlon, cell_maxlon, cell_minlon, cell_minlon]
            ys = [cell_maxlat, cell_maxlat, cell_minlat, cell_minlat, cell_maxlat]
            plt.plot(xs,ys, 'k')
            grid_id = grid.get_bin_idx(((cell_maxlon+cell_minlon)/2), ((cell_maxlat+cell_minlat)/2))
            plt.annotate(str(grid_id), (((cell_maxlon+cell_minlon)/2), ((cell_maxlat+cell_minlat)/2)), ha='center', va='center')

########
#Write shapefile from bins
########
for col in range(grid.nlon):
    for row in range(grid.nlat):
        if grid.bin_idx[col, row] > 0:
            cell_minlon = grid.min_lon+(col)*grid.cell_size
            cell_maxlon = grid.min_lon+(col+1)*grid.cell_size
            cell_maxlat = grid.min_lat+(row+1)*grid.cell_size
            cell_minlat = grid.min_lat+(row)*grid.cell_size
            xs = [cell_minlon, cell_maxlon, cell_maxlon, cell_minlon, cell_minlon]
            ys = [cell_maxlat, cell_maxlat, cell_minlat, cell_minlat, cell_maxlat]
            grid_id = grid.get_bin_idx(((cell_maxlon+cell_minlon)/2), ((cell_maxlat+cell_minlat)/2))
            w.poly([[[xs[0], ys[0]],[xs[1],ys[1]],[xs[2],ys[2]],[xs[3],ys[3]]]])            
            w.record(grid_id)

w.close()

def write_shape(site):
    llon = grid.bin_corners(site[0],site[1])[0]
    rlon = grid.bin_corners(site[0],site[1])[1]
    blat = grid.bin_corners(site[0],site[1])[2]
    tlat = grid.bin_corners(site[0],site[1])[3]
    w.poly([[[llon, tlat], [rlon, tlat], [rlon, blat], [llon, blat]]])
    w.record('oops', round(site[0],1), round(site[1],1))




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
            start_bin_idx = grid.get_bin_idx(x[i][0].x, x[i][0].y) +1
            final_bin_idx = grid.get_bin_idx(x[i][1].x, x[i][1].y) +1
            sfbins.append([start_bin_idx, final_bin_idx])
        #print(len(sfbins))
        for i in sfbins:
            if i[0] != 0:
                if i[1] == -1:
                    matrix[i[0]-1, -1] += 1
                else:
                    matrix[i[0]-1, i[1]-1] += 1
        #print(len(matrix))
    #for i in range(len(matrix)):
    #    #print(i)
    #    if sum(matrix[i]) > 0:
    #        matrix[i] = matrix[i]/sum(matrix[i])
    #matrix = matrix[:,:-1]
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
def plot_sfpoints(sfpoints, num):
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
            if i[2] == 'seeded_on_land':
                plt.plot(fx, fy, 'bo')
            if i[2] == 'outside':
                plt.plot(fx, fy, 'yo')
            if i[2] == 'active':
                plt.plot(fx, fy, 'ro')
            if i[2] == 'home_sweet_home':
                plt.plot(fx, fy, 'mo')
            if i[2] == 'died':
                plt.plot(fx, fy, 'ko') 
    m = Basemap(resolution= 'h', llcrnrlon = 162, llcrnrlat = -51, urcrnrlon = 186, urcrnrlat = -32)
    m.drawcoastlines()
    plt.show()



#############
#just for fun
#############
#get only 14 sites
just14 = bigboy[:, [30, 80, 81, 157, 173, 187, 199, 235, 247, 249, 271, 316, 432, 442]]
just14 = just14[[30, 80, 81, 157, 173, 187, 199, 235, 247, 249, 271, 316, 432, 442], :]

just19 = bigboy19[:, [30, 49, 80, 81, 92, 148, 157, 173, 207, 235, 241, 247, 249, 272, 309, 374, 405, 432, 442]]
just19 = just19[[30, 49, 80, 81, 92, 148, 157, 173, 207, 235, 241, 247, 249, 272, 309, 374, 405, 432, 442], :]

just22 = bigboy22[:, [30, 49, 80, 81, 92, 148, 157, 173, 207, 235, 241, 247, 249, 272, 309, 350, 368, 374, 405, 432, 442, 499]]
just22 = just22[[30, 49, 80, 81, 92, 148, 157, 173, 207, 235, 241, 247, 249, 272, 309, 350, 368, 374, 405, 432, 442, 499], :]

site_bins14 = {'FIO':30, 'BGB':80, 'HSB':81, 'TIM':157, 'LWR':173, 'WEST':187, 'FLE':200, 'TAS':235, 'OPO':247, 'GOB':249, 'KAI':271, 'CAM':316, 'MAU':432, 'CAP':442}
site_bins19 = {'FIO':30, 'RIV':49, 'BGB':80, 'HSB':81, 'JAB':92, 'NMC':148, 'TIM':157, 'LWR':173, 'GOL':207, 'TAS':235, 'HOU':241, 'OPO':247, 'GOB':249, 'KAT':272, 'POG':309, 'PAK':374, 'TEK':405, 'MAU':432, 'CAP':442}
site_bins22 = {'FIO':30, 'RIV':49, 'BGB':80, 'HSB':81, 'JAB':92, 'NMC':148, 'TIM':157, 'LWR':173, 'GOL':207, 'TAS':235, 'HOU':241, 'OPO':247, 'GOB':249, 'KAT':272, 'POG':309, 'RUA': 350, 'DAB': 368, 'PAK':374, 'TEK':405, 'MAU':432, 'CAP':442, 'HIC': 499}

'FIO' = 0
'RIV' = 1
'BGB' = 2
'HSB' = 3
'JAB' = 4
'NMC' = 5
'TIM' = 6
'LWR' = 7
'GOL' = 8
'TAS' = 9
'HOU' = 10
'OPO' = 11
'GOB' = 12
'KAT' = 13
'POG' = 14
'RUA' = 15
'DAB' = 16
'PAK' = 17
'TEK' = 18
'MAU' = 19
'CAP' = 20
'HIC' = 21



#site_order14 = [8, 12, 5, 6, 7, 4, 13, 11, 10, 9, 3, 2, 1, 0]
#site_order19 = [11, 10, 15, 16, 17, 13, 9, 8, 7, 5, 14, 18, 12, 6, 3, 2, 1, 0, 4]
site_order22 = [15, 11, 10, 17, 18, 19, 21, 13, 14, 9, 8, 7, 5, 20, 16, 12, 6, 3, 2, 1, 0, 4]
just14_order = just14[site_order14]
just14_order = just14_order[:,site_order14]
#just19_order = just19[site_order19]
#just19_order = just19_order[:,site_order19]
just22_order = just22[site_order22]
just22_order = just22_order[:,site_order22]

site_labs14 = ['FIO', 'BGB', 'HSB', 'TIM', 'LWR', 'WEST', 'FLE', 'TAS', 'OPO', 'GOB', 'KAI', 'CAM', 'MAU', 'CAP']
site_labs19 = ['FIO', 'RIV', 'BGB', 'HSB', 'JAB', 'NMC', 'TIM', 'LWR', 'GOL', 'TAS', 'RUA', 'OPO', 'GOB', 'KAT', 'POG', 'PAK', 'TEK', 'MAU', 'CAP']
site_labs22 = ['FIO', 'RIV', 'BGB', 'HSB', 'JAB', 'NMC', 'TIM', 'LWR', 'GOL', 'TAS', 'HOU', 'OPO', 'GOB', 'KAT', 'POG', 'RUA', 'DAB', 'PAK', 'TEK', 'MAU', 'CAP', 'HIC']
site_labs = np.asarray(site_labs22)
site_labs = site_labs[site_order22]
df = pd.DataFrame(data = just22_order, index = site_labs, columns = site_labs)
num=np.sum(bigboy[30])
df_pct=(df/num)*100
df_log=np.log10(df_pct)
df_log[np.isneginf(df_log)] = -5

ax = sns.heatmap(df_log, cbar_kws={'label': '% Succesful migrants)'})#, annot=True, fmt='.0f')
cbar = ax.collections[0].colorbar
cbar.set_ticks([1,0,-1,-2,-3])
cbar.set_ticklabels([10,1,.1,.01,.001])
ax.invert_xaxis()
#ax.set_xlabel("Destination Population")
#ax.set_ylabel("Source Population")
#ax.set(xlabel='common xlabel', ylabel='common ylabel')
plt.xlabel("Destination Population")
plt.ylabel("Source Population")
#plt.title("Ocean Connectivity")
plt.tight_layout()
plt.show()


#############
#identify outlier events
#############
start_site='CAP'
end_site='KAI'
end_bin=site_bins[end_site]
total_particles=0
particles_per_month=0
number_of_months=0
s = 's'

def wanderers(start_site, end_site):
    print('mm yyyy: #particles')
    total_particles=0
    particles_per_month=0
    number_of_months=0
    s = 's'
    for file in sorted(glob.glob(f'bigboy/*{start_site}*')):
        new_month=True
        new_month_print=False
        particles_per_month=0
        pts = customout_to_startfinal_points(file)
        for part in pts:
            if grid.get_bin_idx(part[1].x, part[1].y) == site_bins[end_site]:
                total_particles+=1
                particles_per_month+=1
                if new_month:
                    number_of_months+=1
                    new_month=False
                    new_month_print=True
        if new_month_print:
            print(f'{file[-6:-4]} {file[-10:-6]}: {particles_per_month}')
    print(f'{total_particles} particle{s*(total_particles>1)} made it from {start_site} to {end_site} over {number_of_months} unique months')



#############
#plot reef
#############
sf = shp.Reader("NZ_rocky_reef.shp")
for shape in sf.shapeRecords():
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    plt.plot(x,y)


#exclude empty rows
nonemptyr = 0
for i in outmat:
    if sum(i) > 0:
        nonemptyr += 1

reefless_bins = [49, 102, 116, 118, 120, 122, 130, 132, 134, 139, 143, 145, 166, 174, 200, 201, 217, 227, 208, 207, 215, 216, 235, 236, 391, 390, 389, 384, 383, 336, 337, 341, 342, 343, 328, 287, 281, 275, 268, 269, 220, 474, 472, 471, 473, 475, 477, 479, 192, 188, 186, 176, 167, 165, 163, 151, 150, 149, 147]

matrixr = np.empty((nonemptyr, 532))

count = 0
for i in range(len(outmat)):
    if sum(outmat[i] > 0):
        matrixr[count] = outmat[i]
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



for file in glob.glob('/nesi/nobackup/vuw03073/bigboy/all_settlement/'):
    sfpoints = (nc_to_startfinal_points(file))


site_order14 = [0, 1, 5, 4, 3, 6, 2, 13, 12, 11, 10, 9, 8, 7]#[::-1]
site_labs14 = ['OPO', 'MAU', 'CAP', 'TAS', 'FLE', 'WEST', 'LWR', 'FIO', 'BGB', 'HSB', 'TIM', 'GOB', 'KAI', 'CAM']



just14 = mat1[:, [39, 124, 187, 230, 235, 239, 246, 288, 346, 347, 374, 390, 395, 402]]
just14 = just14[[39, 124, 187, 230, 235, 239, 246, 288, 346, 347, 374, 390, 395, 402], :]

site_labs = np.asarray(site_labs14)
site_labs = site_labs[site_order14]
df = pd.DataFrame(data = just14_order, index = site_labs, columns = site_labs)
num=np.sum(mat1[39])
df_pct=(df/num)*100
df_log=np.log10(df_pct)

ax = sns.heatmap(df_log, cbar_kws={'label': '% Succesful migrants)'})#, annot=True, fmt='.0f')
cbar = ax.collections[0].colorbar
cbar.set_ticks([1,0,-1,-2,-3])
cbar.set_ticklabels([10,1,.1,.01,.001])
ax.invert_xaxis()
#ax.set_xlabel("Destination Population")
#ax.set_ylabel("Source Population")
#ax.set(xlabel='common xlabel', ylabel='common ylabel')
plt.xlabel("Destination Population")
plt.ylabel("Source Population")
#plt.title("Ocean Connectivity")
plt.tight_layout()
plt.show()












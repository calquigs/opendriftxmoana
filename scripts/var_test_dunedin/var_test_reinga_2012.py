#!/usr/bin/env python
# 
import os
import sys
import time
import numpy as np
import matplotlib
from datetime import datetime, timedelta
from opendrift.readers import reader_ROMS_native_MOANA
from opendrift.models.bivalvelarvae import BivalveLarvae

start_month = int(sys.argv[1])
months = ['201201','201202','201203','201204','201205','201206','201207','201208','201209','201210','201211','201212']
###############################
# MODEL SELECTION
###############################
o = BivalveLarvae(loglevel=0)#,logfile='mussel_forwardtrack_%s_%s.log' % (year,month) )  # Set loglevel to 0 for debug information
###############################
# READERS
###############################



path201201 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201201.nc'
path201202 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201202.nc'
path201203 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201203.nc'
path201204 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201204.nc'
path201205 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201205.nc'
path201206 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201206.nc'
path201207 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201207.nc'
path201208 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201208.nc'
path201209 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201209.nc'
path201210 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201210.nc'
path201211 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201211.nc'
path201212 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201212.nc'
path201301 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201301.nc'
path201302 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201302.nc'


paths = [path201201, path201202, path201203, path201204, path201205, path201206, path201207, path201208, path201209, path201210, path201211, path201212, path201301, path201302]

# reader_moana_dec15 = reader_ROMS_native_MOANA.Reader(data_path+"nz5km_his_201707.nc") # load data for that year
# reader_moana_dec15.multiprocessing_fail = True # thisb ypasses the use of multi core for coordinates conversion and seems to make the model run much faster.


reader_moana_0 = reader_ROMS_native_MOANA.Reader(paths[start_month]) # load data for that year
reader_moana_1 = reader_ROMS_native_MOANA.Reader(paths[start_month+1]) # load data for that year
reader_moana_2 = reader_ROMS_native_MOANA.Reader(paths[start_month+2]) # load data for that year
reader_moana_0.multiprocessing_fail = True # bypass the use of multi core for coordinates conversion and seems to make the model run much faster.
reader_moana_1.multiprocessing_fail = True # bypass the use of multi core for coordinates conversion and seems to make the model run much faster.
reader_moana_2.multiprocessing_fail = True # bypass the use of multi core for coordinates conversion and seems to make the model run much faster.

# # Making customised landmask - not required here, we are using the ROMS landmask i.e. included in netcdf files
# reader_landmask = reader_global_landmask.Reader(
#                     llcrnrlon=171.0, llcrnrlat=184.5,
#                     urcrnrlon=-42.0, urcrnrlat=-34.0)

# use native landmask of ROMS files
o.add_reader([reader_moana_0, reader_moana_1, reader_moana_2]) # 
o.set_config('general:use_auto_landmask', False) # prevent opendrift from making a new dynamical landmask

###############################
# PARTICLE SEEDING
###############################
#Dunedin
lons = [170.75, 170.8, 170.8, 170.75]
lats = [-45.95, -45.95, -46, -46]

def create_seed_times(start, end, delta):
  """
  crate times at given interval to seed particles
  """
  out = []
  start_t = start
  end_t = datetime.strptime(str(end), "%Y-%m-%d %H:%M:%S")
  while start_t < end:
    out.append(start_t) 
    start_t += delta
  return out

times = create_seed_times(reader_moana_0.start_time, 
                          reader_moana_0.end_time, timedelta(hours = 2))


number = 100
z = np.random.uniform(-10,0,size=len(times)) # generate random depth

for i in range(len(times)):
  o.seed_within_polygon(lons, lats, number = number, time = times[i], z = z[i])

###############################
# PHYSICS
###############################
# these will list all possible options for that model
o.set_config('environment:fallback:x_wind', 0.0)
o.set_config('environment:fallback:y_wind', 0.0)
o.set_config('environment:fallback:x_sea_water_velocity', 0.0)
o.set_config('environment:fallback:y_sea_water_velocity', 0.0)
o.set_config('environment:fallback:sea_floor_depth_below_sea_level', 100000.0)

# No diffusion - constant in that example
#Kxy = 1.0 #0.1176 # m2/s-1
#Kz = 0.001 # m2/s-1
Kxy = 0.1176  #0.1176 # m2/s-1
Kz = 0.01# m2/s-1
o.set_config('drift:horizontal_diffusivity',Kxy) # using new config rather than current uncertainty
o.set_config('environment:fallback:ocean_vertical_diffusivity', Kz) # specify constant ocean_vertical_diffusivity in m2.s-1
# seed
o.set_config('seed:ocean_only',True) # keep only particles from the "frame" that are on the ocean
# drift
o.set_config('drift:advection_scheme','runge-kutta4') # or 'runge-kutta'
o.set_config('drift:current_uncertainty', 0.0 ) # note current_uncertainty can be used to replicate an horizontal diffusion spd_uncertain = sqrt(Kxy*2/dt)  
o.set_config('drift:max_age_seconds', 3600*24*30) # 
o.set_config('drift:lift_to_seafloor',True)
#processes
o.set_config('drift:vertical_mixing', False)  
o.set_config('general:coastline_action','previous') # option('none', 'stranding', 'previous', default='stranding')
#o.set_config('drift:min_settlement_age_seconds', 3600*24*21) #

o.list_config()
# o.list_configspec()

###############################
# RUN 
###############################
# Running model (until end of driver data)

lons_start = o.elements_scheduled.lon
lats_start = o.elements_scheduled.lat

o.run(stop_on_error=False,
      end_time=reader_moana_2.end_time,
      time_step=900, 
      time_step_output = 86400.0,
      export_variables = [])

index_of_first, index_of_last = o.index_of_activation_and_deactivation()
lons = o.get_property('lon')[0]
lats = o.get_property('lat')[0]
status = o.get_property('status')[0]
lons_end = lons[index_of_last, range(lons.shape[1])]
lats_end = lats[index_of_last, range(lons.shape[1])]
status_end = status[index_of_last, range(lons.shape[1])]


outFile = open(f'variability_test_reinga_{months[start_month]}.txt','w')

for i in range(len(lons_end)):
  outFile.write(str(lons_start[i])+","+str(lats_start[i])+","+str(lons_end[i])+","+str(lats_end[i])+","+str(status_end[i])+"\n")

outFile.close()





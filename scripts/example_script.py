#!/usr/bin/env python

import os
import sys
import time
import numpy as np
import matplotlib
from datetime import datetime, timedelta
from opendrift.readers import reader_ROMS_native_MOANA #I believe this is a custom reader that Simon Weppe has written, I'm not sure if the basemodel's reader_ROMS_native can handle the backbone files?
from opendrift.models.bivalvelarvae import BivalveLarvae #Bivalvelarvae.py is a custom module that Simon Weppe has written that includes some useful features, but you can replace this with whichever model you want to run
#e.g.
#from opendrift.models.pelagicegg import PelagicEggDrift


##############################
#CREATING THE SIMLATION OBJECT
##############################
o = BivalveLarvae(loglevel = 0) #loglevel = 0 will print out what the model is doing at every step, so helpful when getting started, but creates huge output files. When you are running larger quantities of runs, you'll want to change to loglevel = 30

#provide paths to the data for the months you want
data_path0 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201606.nc' 
data_path1 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201607.nc' 

#create readers for the backbone files
reader0 = reader_ROMS_native_MOANA.Reader(data_path0)
reader1 = reader_ROMS_native_MOANA.Reader(data_path1)

reader0.multiprocessing_fail = True #No idea what these lines do, but Simon told me that they make the model ruun faster so I've always included them
reader1.multiprocessing_fail = True 

#add readers to the simulation
o.add_reader([reader0, reader1])


##############################
#SET CONFIGURATIONS
##############################

#There are a bunch of configs you can set, I'll just explain the ones I've used

o.set_config('general:use_auto_landmask', False) #If you use True, opendrift will automatically create a higher-res landmask and project data to try and fill in the gaps. False runs using the backbone's block, 5km landmask.
# setting fallback values
o.set_config('environment:fallback:x_wind', 0.0)
o.set_config('environment:fallback:y_wind', 0.0)
o.set_config('environment:fallback:x_sea_water_velocity', 0.0)
o.set_config('environment:fallback:y_sea_water_velocity', 0.0)
o.set_config('environment:fallback:sea_floor_depth_below_sea_level', 100000.0)

#setting diffusion— some horizontal and vertical randomization
Kxy = 0.1176  #m2/s-1. These numbers come from some calculation based on the resolution of the model, some one calculated this a while ago for the backbone
Kz = 0.01 #m2/s-1
o.set_config('drift:horizontal_diffusivity',Kxy) 
o.set_config('environment:fallback:ocean_vertical_diffusivity', Kz) 

o.set_config('seed:ocean_only', False) #if True, any particles seeded on land will try and auto move to the water, if False they will just be retired. 

o.set_config('drift:advection_scheme','runge-kutta4') #some sort of math thing that I don't understand. Either 'runge-kutta4' or 'runge-kutta' should work. 
o.set_config('drift:vertical_mixing', False) #tbh idk what this does


#tell particles what to do when they hit the coast/floor
o.set_config('general:seafloor_action', 'lift_to_seafloor')
o.set_config('general:coastline_action','previous')

#decide how long you want your particles to float. min_settlement_age is a feature of BivalveLarvae
o.set_config('drift:max_age_seconds', 3600*24*35)
o.set_config('drift:min_settlement_age_seconds', 3600*24*21)

o.list_config() #print out all your configs just for fun/help debugging


##############################
#SEED PARTICLES
##############################
#There's a bunch of different ways to seed particles, here's my go-to method
#give corners of a polygon to seed within
minlon = 173.2
maxlon = 173.3
minlat = -35.6
maxlat = -35.5
lons = [minlon, maxlon, maxlon, minlon]
lats = [maxlat, maxlat, minlat, minlat]

time0 = reader0.start_time
z = -5
number = 100

o.seed_within_polygon(lons, lats, number = number, z = z, time = time0)

##############################
#RUN
##############################
#a bunch of tricks you can do here to save only exactly what you want, but here's the basics

o.run(stop_on_error=False,
      end_time=reader1.end_time,
      time_step=3600, 
      time_step_output = 3600.0,
      outfile= 'path/outfile_name.nc')

o.plot(filename = 'path/figure_name.jpg')
o.animation(filename = 'path/anim_name.mp4')



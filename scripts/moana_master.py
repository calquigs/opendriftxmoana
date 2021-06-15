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

###############################
#Parse arguments
###############################
import argparse

parser = argparse.ArgumentParser(description='Run opendrift simulation')
parser.add_argument('-i', '--input', type=str, required=True, help='Input reaader file path (ending in /)')
parser.add_argument('-o', '--output', type=str, help='Output file path (ending in /')
parser.add_argument('-n', '--name', type=str, default='opendrift', help='Output file name')
parser.add_argument('-ym', '--yearmonth', type=int, required=True, help='Month and year to seed (yyyymm)')
parser.add_argument('-lon', '--upperleftlon', type=float, required=True, help='Longitude of upper left hand corner of .05 deg bin to seed')
parser.add_argument('-lat', '--upperleftlat', type=float, required=True, help='Latitude of upper left hand corner of .05 deg bin to seed')

args = parser.parse_args()

###############################
#Create Readers
###############################
yyyymm = str(args.yearmonth)
mm = int(yyyymm[-2:])
yyyy = int(yyyymm[0:4])
date0 = datetime(yyyy, mm, 1)
mm0 = date0.strftime('%m')
yyyy0 = date0.strftime('%Y')
date1 = date0 + timedelta(days = 30)
mm1 = date1.strftime('%m')
yyyy1 = date1.strftime('%Y')
date2 = date1 + timedelta(days = 30)
mm2 = date2.strftime('%m')
yyyy2 = date2.strftime('%Y')

path0 = args.input + 'nz5km_his_' + yyyy0 + mm0 + '.nc'
path1 = args.input + 'nz5km_his_' + yyyy1 + mm1 + '.nc'
path2 = args.input + 'nz5km_his_' + yyyy2 + mm2 + '.nc'

reader0 = reader_ROMS_native_MOANA.Reader(path0)
reader1 = reader_ROMS_native_MOANA.Reader(path1)
reader2 = reader_ROMS_native_MOANA.Reader(path2)

reader0.multiprocessing_fail = True 
reader1.multiprocessing_fail = True 
reader2.multiprocessing_fail = True 


###############################
#Create Simulation Object
###############################

o = BivalveLarvae(loglevel=0)
o.add_reader([reader0, reader1, reader2])

###############################
#Seed Particles
###############################

ullon = args.upperleftlon 
ullat = args.upperleftlat

lons = [ullon, ullon+.05, ullon+.05, ullon]
lats = [ullat, ullat, ullat-.05, ullat-.05]


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

times = create_seed_times(reader0.start_time, 
                          reader0.end_time, timedelta(hours = 2))


number = 100
z = np.random.uniform(-10,0,size=len(times)) # generate random depth

for time in times:
  o.seed_within_polygon(lons, lats, number = number, time = time, z = z[i])


###############################
#Set Configs
###############################
o.set_config('general:use_auto_landmask', False)
o.set_config('environment:fallback:x_wind', 0.0)
o.set_config('environment:fallback:y_wind', 0.0)
o.set_config('environment:fallback:x_sea_water_velocity', 0.0)
o.set_config('environment:fallback:y_sea_water_velocity', 0.0)
o.set_config('environment:fallback:sea_floor_depth_below_sea_level', 100000.0)

Kxy = 0.1176  #m2/s-1
Kz = 0.01 #m2/s-1

o.set_config('drift:horizontal_diffusivity',Kxy) 
o.set_config('environment:fallback:ocean_vertical_diffusivity', Kz) 
o.set_config('seed:ocean_only',True)
o.set_config('drift:advection_scheme','runge-kutta4')
o.set_config('drift:current_uncertainty', 0.0)
o.set_config('drift:max_age_seconds', 3600*24*30)
o.set_config('drift:lift_to_seafloor',True)
o.set_config('drift:vertical_mixing', False)
o.set_config('general:coastline_action','previous')

o.list_config()

###############################
#Run
###############################

lons_start = o.elements_scheduled.lon
lats_start = o.elements_scheduled.lat

o.run(stop_on_error=False,
      end_time=reader2.end_time,
      time_step=900, 
      time_step_output = 86400.0,
      export_variables = [],
      file_name = )

index_of_first, index_of_last = o.index_of_activation_and_deactivation()
lons = o.get_property('lon')[0]
lats = o.get_property('lat')[0]
status = o.get_property('status')[0]
lons_end = lons[index_of_last, range(lons.shape[1])]
lats_end = lats[index_of_last, range(lons.shape[1])]
status_end = status[index_of_last, range(lons.shape[1])]

outFile = open(f'{args.output}{args.name}_{yyyy0}{mm0}.txt','w')

for i in range(len(lons_end)):
  outFile.write(str(lons_start[i])+","+str(lats_start[i])+","+str(lons_end[i])+","+str(lats_end[i])+","+str(status_end[i])+"\n")

outFile.close()

o.animation



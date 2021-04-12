#!/usr/bin/env python
"""
Bivalve Larvae class
=============
"""

from opendrift.readers import reader_ROMS_native_MOANA
from datetime import datetime,timedelta
from opendrift.models.bivalvelarvae import BivalveLarvae
import numpy as np

o = BivalveLarvae(loglevel=0)

data_path0 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201512.nc'
data_path1 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201601.nc'
data_path2 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201602.nc'
data_path3 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201603.nc'
data_path4 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201604.nc'
data_path5 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201605.nc'
data_path6 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201606.nc'

reader_moana_dec15 = reader_ROMS_native_MOANA.Reader(data_path0) # load data for that month
reader_moana_jan16 = reader_ROMS_native_MOANA.Reader(data_path1) # load data for that month
reader_moana_feb16 = reader_ROMS_native_MOANA.Reader(data_path2) # load data for that month
reader_moana_mar16 = reader_ROMS_native_MOANA.Reader(data_path3) # load data for that month
reader_moana_apr16 = reader_ROMS_native_MOANA.Reader(data_path4) # load data for that month
reader_moana_may16 = reader_ROMS_native_MOANA.Reader(data_path5) # load data for that month
reader_moana_jun16 = reader_ROMS_native_MOANA.Reader(data_path6) # load data for that month
reader_moana_dec15.multiprocessing_fail = True # bypass the use of multi core for coordinates conversion and seems to make the model run much faster.
reader_moana_jan16.multiprocessing_fail = True # bypass the use of multi core for coordinates conversion and seems to make the model run much faster.
reader_moana_feb16.multiprocessing_fail = True # bypass the use of multi core for coordinates conversion and seems to make the model run much faster.
reader_moana_mar16.multiprocessing_fail = True # bypass the use of multi core for coordinates conversion and seems to make the model run much faster.
reader_moana_apr16.multiprocessing_fail = True # bypass the use of multi core for coordinates conversion and seems to make the model run much faster.
reader_moana_may16.multiprocessing_fail = True # bypass the use of multi core for coordinates conversion and seems to make the model run much faster.
reader_moana_jun16.multiprocessing_fail = True # bypass the use of multi core for coordinates conversion and seems to make the model run much faster.

o.add_reader([reader_moana_dec15, reader_moana_jan16, reader_moana_feb16, reader_moana_mar16, reader_moana_apr16, reader_moana_may16, reader_moana_jun16]) # [reader_landmask,reader_moana_dec15]


#configs
o.set_config('environment:fallback:ocean_vertical_diffusivity',0.001) # m2/s

o.set_config('general:coastline_action', 'previous')
o.set_config('drift:lift_to_seafloor',True)

o.set_config('drift:vertical_advection', True)
o.set_config('drift:vertical_mixing', True)
o.set_config('drift:min_settlement_age_seconds', 3600) # minimum age before settling can occur
o.set_config('drift:max_age_seconds', 3600*24*42) # 

o.set_config('vertical_mixing:diffusivitymodel', 'constant') # use eddy diffusivity from ocean model, or fallback value
o.set_config('vertical_mixing:timestep', 900.) # seconds - # Vertical mixing requires fast time step  (but for constant diffusivity, use same as model step)


# Calvin's PhD thesis

## Running OpenDrift scripts

### Installing dependencies


We recommend using a virtual environment to install all dependencies for this project. [miniconda](https://docs.conda.io/en/latest/miniconda.html) is a good way to create a virtual environment with specific versions of Python, working on all major plateforms (Windows, MacOs and Linux).

After installing miniconda, you need to install [OpenDrift (forked version)](https://github.com/simonweppe/opendrift):
```
git clone https://github.com/simonweppe/opendrift
cd opendrift
conda env create -f environment.yml
conda activate opendrift
pip install -e .
```

Everytime you want to work on the project, you need to *activate* it before: 
```
conda activate opendrift
```
You should see `(opendrift)` prepended to your command line prompt as a hint that it is now activated.

To deactivate the environment, type
```
conda deactivate
```

## How to create a simulation

Create your simulation object,
```
o = BivalveLarvae(loglevel=0)
```
add readers,
```
data_path0 = '/nesi/nobackup/mocean02574/NZB_N50/nz5km_his_201512.nc'
reader_moana_dec15 = reader_ROMS_native_MOANA.Reader(data_path0)
reader_moana_dec15.multiprocessing_fail = True # bypass the use of multi core for coordinates conversion and seems to make the model run much faster.
o.add_reader([reader_moana_dec15])
```
and specify a land mask to use.
```
o.set_config('general:use_auto_landmask', False) # this will use the built in landmask in the Moana reader.
```

### Seeding Particles
Define the corners of the polygon you want to seed within.
```
#Cape Reinga
lons = [170.75, 170.8, 170.8, 170.75]
lats = [-45.95, -45.95, -46, -46]
```
Define number of particles, depth, and time of release.
```
number = 100
z = -10
time = datetime(2015,12,1)
```

Seed particles
```
o.seed_within_polygon(lons, lats, number = number, time = time, z = z)
```

### Set configs
There's a ton of different configurations you can add to modify particle+environmental properties
```
o.set_config('drift:max_age_seconds', 3600*24*30) 
o.set_config('drift:lift_to_seafloor',True)
o.set_config('drift:horizontal_diffusivity',.01) 
o.set_config('environment:fallback:ocean_vertical_diffusivity', .01) 
#etc.
```

### Run model
Use `o.run()` to run the model. 
```
o.run(stop_on_error=False, 
    end_time=reader_moana_dec15.end_time, 
    time_step=900, outfile= 'nc_out.nc', 
    time_step_output = 3600.0, 
    export_variables = [])
```

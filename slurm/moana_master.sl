#!/bin/bash -e

#SBATCH --account=vuw03073                      #Type the project code you want to use for this analyis		 $
#SBATCH --job-name=14sites20years          #This willl be the name appear on the queue			 $
#SBATCH --mem=2G                               #Amount of memory you need                                       $
#SBATCH --cpus-per-task=2                       #Amount of CPUs (logical)                                        $
#SBATCH --time=01:30:00                         #Duration dd-hh:mm:ss                                            $
#SBATCH --output=/nesi/project/vuw03073/testScripts/slurmOut/14sites20years_%a.%j.txt    #Name of the output file                                 $
#SBATCH --mail-type=ALL                         #This will send you an email when the STARTS and ENDS		 $
#SBATCH --mail-user=calquigs@gmail.com          #Enter your email address.                                       $
#SBATCH --profile=task
#SBATCH --array=0                     # Array jobs

#SBATCH --export NONE

export SLURM_EXPORT_ENV=ALL

#purging any modules/software loaded in the background prior to submitting script
module purge

#Load the required module for analysis/simulation
module load Miniconda3
source activate opendrift_simon

#Set variables
inPath='/nesi/nobackup/mocean02574/NZB_N50/'
outPath='/nesi/project/vuw03073/testScripts/bigboy_test/'
name='GOB' 
lon=173.3
lat=-42.9

#names=('OPO' 'MAU' 'CAP' 'WEST' 'FLE' 'TAS' 'CAM' 'LWR' 'KAI' 'GOB' 'TIM' 'FIO' 'HSB' 'BGB')
#lons=(173.2 176.0 176.3 172.4 172.7 173.1 174.3 171.9 173.7 173.3 171.3 166.8 168.2 168.2)
#lats=(-35.5 -37.4 -40.9 -40.5 -40.5 -41.1 -41.7 -41.3 -42.4 -42.9 -44.4 -45.1 -46.8 -46.9)

#names=('OPO' 'MAU' 'CAP')
#lons=(173.2 176.0 176.3)
#lats=(-35.5 -37.4 -40.9)

#names=('WEST' 'FLE' 'TAS')
#lons=(172.4 172.7 173.1)
#lats=(-40.5 -40.5 -41.1)

#names=('CAM' 'LWR' 'KAI')
#lons=(174.3 171.9 173.7)
#lats=(-41.7 -41.3 -42.4)

#names=('GOB' 'TIM' 'FIO')
#lons=(173.3 171.3 166.8)
#lats=(-42.9 -44.4 -45.1)

#names=('HSB' 'BGB')
#lons=(168.2 168.2)
#lats=(-46.8 -46.9)

#Create array of yyyymm
months=(08) #(01 02 03 04 05 06 07 08 09 10 11 12)
#years=($(seq 1994 2016))
years=(2004)
declare -a ym

for i in "${months[@]}"
do
    for j in "${years[@]}"
    do
         ym+=("$j""$i")
    done
done

num_runs_per_site=$((${#months[@]}*${#years[@]}))
#name=${names[$(($SLURM_ARRAY_TASK_ID/$num_runs_per_site))]}
#lon=${lons[$(($SLURM_ARRAY_TASK_ID/$num_runs_per_site))]}
#lat=${lats[$(($SLURM_ARRAY_TASK_ID/$num_runs_per_site))]}

echo $num_runs_per_site
echo $name
echo ${ym[$(($SLURM_ARRAY_TASK_ID%$num_runs_per_site))]}

#run jobs
python /nesi/project/vuw03073/opendriftxmoana/scripts/moana_master.py -i $inPath -o $outPath -n $name -ym ${ym[$(($SLURM_ARRAY_TASK_ID%$num_runs_per_site))]} -lon $lon -lat $lat



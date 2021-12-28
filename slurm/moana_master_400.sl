#!/bin/bash -e

#SBATCH --account=vuw03073                      #Type the project code you want to use for this analyis		 $
#SBATCH --job-name=all_settlement_test          #This willl be the name appear on the queue			 $
#SBATCH --mem=5G                               #Amount of memory you need                                       $
#SBATCH --time=00:10:00                         #Duration dd-hh:mm:ss                                            $
#SBATCH --cpus-per-task=2                       #Amount of CPUs (logical)                                        $
#SBATCH --output=/nesi/project/vuw03073/testScripts/slurmOut/all_settlement_test_%a.%j.txt    #Name of the output file                                 $
#SBATCH --mail-type=ALL                         #This will send you an email when the STARTS and ENDS		 $
#SBATCH --mail-user=calquigs@gmail.com          #Enter your email address.                                       $
#SBATCH --profile=task
#SBATCH --array=0-1                     # Array jobs
#SBATCH --export NONE

export SLURM_EXPORT_ENV=ALL
export HDF5_USE_FILE_LOCKING=FALSE

#purging any modules/software loaded in the background prior to submitting script
module purge

#Load the required module for analysis/simulation
module load Miniconda3
source activate opendrift_simon

#Set variables
inPath='/nesi/nobackup/mocean02574/NZB_3/'
outPath='/nesi/nobackup/vuw03073/bigmomma/'

#14pops
#regions=('taranaki' 'waikato' '90milebeach' 'northland' 'hauraki' 'bay_o_plenty' 'east_cape' 'hawkes_bay' 'wairarapa' 'wellington' 'marlborough' 'kahurangi' 'west_coast' 'fiordland' 'southland' 'stewart_island' 'otago' 'canterbury' 'kaikoura' 'chatham' 'auckland_isl')
regions=('taranaki')

#Create array of yyyymm
months=(01 02 03 04 05 06 07 08 09 10 11 12)
#months=(08)
#years=($(seq 1994 2016))
years=($(seq 1994 1996))
#years=(1994)
declare -a ym

for i in "${months[@]}"
do
    for j in "${years[@]}"
    do
         ym+=("$j""$i")
    done
done

num_runs_per_site=$((${#months[@]}*${#years[@]}))
region=${regions[$(($SLURM_ARRAY_TASK_ID/$num_runs_per_site))]}
lon=${lons[$(($SLURM_ARRAY_TASK_ID/$num_runs_per_site))]}
lat=${lats[$(($SLURM_ARRAY_TASK_ID/$num_runs_per_site))]}

echo $num_runs_per_site
echo $region
echo ${ym[$(($SLURM_ARRAY_TASK_ID%$num_runs_per_site))]}

#run jobs
python /nesi/project/vuw03073/opendriftxmoana/scripts/moana_master_400.py -i $inPath -o $outPath -r $region -ym ${ym[$(($SLURM_ARRAY_TASK_ID%$num_runs_per_site))]}


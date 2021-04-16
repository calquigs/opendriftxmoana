#!/bin/bash -e

#SBATCH --account=vuw03073                      #Type the project code you want to use for this analyis		 $
#SBATCH --job-name=variability_reinga          #This willl be the name appear on the queue			 $
#SBATCH --mem=20G                               #Amount of memory you need                                       $
#SBATCH --cpus-per-task=2                       #Amount of CPUs (logical)                                        $
#SBATCH --time=04:00:00                         #Duration dd-hh:mm:ss                                            $
#SBATCH --output=/nesi/project/vuw03073/testScripts/slurmOut/variability_reinga_%a.%j.txt    #Name of the output file                                 $
#SBATCH --mail-type=ALL                         #This will send you an email when the STARTS and ENDS		 $
#SBATCH --mail-user=calquigs@gmail.com          #Enter your email address.                                       $
#SBATCH --profile=task
#SBATCH --array=0-11                     # Array jobs

#SBATCH --export NONE

export SLURM_EXPORT_ENV=ALL

#purging any modules/software loaded in the background prior to submitting script
module purge

#Load the required module for analysis/simulation
module load Miniconda3
source activate opendrift_simon

#run jobs
#Is this a good way to repeat a job array for multiple scripts? Will the memory/walltime requested in the header apply to 1loop iteration * 1array iteration?
for script in /nesi/project/vuw03073/opendriftxmoana/scripts/var_test_reinga/
do
  python script ${SLURM_ARRAY_TASK_ID}
done



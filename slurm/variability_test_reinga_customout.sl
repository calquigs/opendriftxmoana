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
#SBATCH --array=0-4                     # Array jobs

#SBATCH --export NONE

export SLURM_EXPORT_ENV=ALL

#purging any modules/software loaded in the background prior to submitting script
module purge

#Load the required module for analysis/simulation
module load Miniconda3
source activate opendrift_simon

#create temporary directory for storing output
mkdir run_${SLURM_ARRAY_TASK_ID}
cd run_${SLURM_ARRAY_TASK_ID}

#run jobs
python /nesi/project/vuw03073/opendriftxmoana/variability_test_reinga_customout.py ${SLURM_ARRAY_TASK_ID}

#stitch files
cat variability_test_reinga_* > ../variability_test_reinga_inarray.txt

#exit and delete temporary directory
cd ..
rm -r run_${SLURM_ARRAY_TASK_ID}



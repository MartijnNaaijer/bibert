#!/bin/bash

### Comment lines start with ## or #+space
### Slurm option lines start with #SBATCH

### Here are the SBATCH parameters that you should always consider:
#SBATCH --time=0-10:00:00   ## days-hours:minutes:seconds
#SBATCH --mem 40G         ## 3000 = 3GB ram (hardware ratio is < 4GB/core)
#SBATCH --ntasks=1          ## Not strictly necessary because default is 1
#SBATCH --cpus-per-task=1   ## Use greater than 1 for parallelized jobs
#SBATCH --gpus=1

### Here are other SBATCH parameters that you may benefit from using, currently commented out:
###SBATCH --job-name=hello1 ## job name
###SBATCH --output=job.out  ## standard out file

hostname	## Prints the system hostname
date	 	## Prints the system date

echo load mamba
module load mamba
 
source activate base
# echo activate bert
# conda activate bert4
conda activate test1

# echo load gpu cuda
# module load gpu #cuda/12.6
# module load gpu cuda/12.6
# echo load cudnn
# module load gpu cudnn
echo run script
srun python main.py -auh=False -ahp=0 -adds=False -aus=False -asp=0
echo 'finished'

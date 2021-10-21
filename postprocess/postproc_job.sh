#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --time=3:00:00
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=evaluation.log



module load gcc/7.3.0
module load R/3.6.3
Rscript multi_o3_cv.R 
Rscript multi_pm2.5_cv.R 

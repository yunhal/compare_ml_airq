#!/bin/bash
#SBATCH --job-name=compare_ml_dl          # name of the job
#SBATCH --partition=gpu             # partition to be used (defq, gpu or intel)
#SBATCH --time=48:00:00             # walltime (up to 96 hours)
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks=1                  # number of tasks (i.e. parallel processes) to be started


source  ~/miniconda3/etc/profile.d/conda.sh

conda activate DLair

module load git
module load cuda/11.1


root=/bigdata/casus/atmos/play/DLair/code-compare-ml-dl  #/bigdata/casus/atmos/DL_atm_model/projects/code/code-compare-ml-dl
cd $root

python main.py --wrf_path ./data/WRF --aqs_path ./data/AQS --out_path runoutput

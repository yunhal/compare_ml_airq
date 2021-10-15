#!/bin/bash
#SBATCH --job-name=compare_ml_dl_p65          # name of the job
#SBATCH --partition=defq             # partition to be used (defq, gpu or intel)
#SBATCH --time=00:20:00             # walltime (up to 96 hours)
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks=1                  # number of tasks (i.e. parallel processes) to be started
#SBATCH --cpus-per-task=1

source  ~/miniconda3/etc/profile.d/conda.sh
conda activate DLair
module load git
module load cuda/11.1

# if the input datasets are not under ./data, you must provide the path (see the main.py)
# for example,  python main.py --wrf_path ./data/WRF --aqs_path ./data/AQS 

exp_name="p8split" # train:test split ratio is 0.8 
echo modeloutput_$exp_name
echo plot_$exp_name

root=/bigdata/casus/atmos/play/DLair/code-compare-ml-dl  #/bigdata/casus/atmos/DL_atm_model/projects/code/code-compare-ml-dl
cd $root

python main.py --pipeml 0  --out_path modeloutput_$exp_name --plotpath plot_$exp_name  --pipeploteval 1  --pipeplotimportance 1 

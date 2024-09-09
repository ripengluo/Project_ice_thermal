#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=0.996_200K_lrp
#SBATCH -N 1 -n 24 -t 24:00:00 
#SBATCH --tasks-per-node=24
#SBATCH --mem=64000mb


module purge
module load cairo/1.14.12-gcc-4.8.5
module load python/3.6.4-gcc-4.8.5
module load fftw/3.3.5-icc-14-double
module load intelcompiler/mkl-15
module load intelcompiler/18.0.0
module load gcc/7.2.0

export LD_LIBRARY_PATH=/BIGDATA1/th_sz_kyu_1/programs/lammps-master-16Jan20/lib64/:$LD_LIBRARY_PATH
export PYTHONPATH=/BIGDATA1/th_sz_kyu_1/programs/lammps-master-16Jan20/lib/python3.6/site-packages:/BIGDATA1/th_sz_kyu_1/.local/lib/python3.6/site-packages/:
cd  $SLURM_SUBMIT_DIR
srun -n 24 python3 prog_restart.py > logfile

sleep 1
 

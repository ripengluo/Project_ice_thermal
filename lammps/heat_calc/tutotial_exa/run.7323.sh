#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=example
#SBATCH -N 1 -n 24 -t 24:00:00 -c 1
#SBATCH --tasks-per-node=24
#SBATCH --mem=64000mb

module purge
module load python/3.6.4-gcc-4.8.5
module load fftw/3.3.5-icc-14-double
module load intelcompiler/mkl-15
module load intelcompiler/18.0.0

export LD_LIBRARY_PATH=/BIGDATA1/th_sz_kyu_1/programs/lammps-12Dec18/lib64/:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
cd  $SLURM_SUBMIT_DIR
srun -n 24 /BIGDATA1/th_sz_kyu_1/programs/lammps-12Dec18//bin/lmp_mpi -in example.in > logfile

sleep 1
 

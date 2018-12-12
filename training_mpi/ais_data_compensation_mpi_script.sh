#!/bin/bash -l

# Batch script to run an MPI parallel job with the upgraded software
# stack under SGE with Intel MPI.

#1. Force bash as the executing shell.
#$ -S /bin/bash

#2. Request one hour of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=8:00:0

#3. Request 8 gigabyte of RAM per procee (must be an integer)
#$ -l mem=8G

#4. Request 15 gigabyte of TMPDIR space per node (default is 10GB)
#$ -l tmpfs=15G

#5. Set the name of the job
#$ -N AIS_trajectory_compensation_mpi_script

#6. Select the MPI parallel environment and 16 processes.
#$ -pe mpi 16

# 7. Set the working directory to somewhere in your scratch space.  This is
# a necessary step with the upgraded software stack as compute nodes cannot
# write to $HOME.
#$ -wd /home/ucesxc0/Scratch/output/ais_trajectory_compensation_mpi

# 8. Run our MPI job.  GERun is a wrapper that launches MPI jobs on our clusters.
module load python3/recommended
gerun ./image_based_trajectory_CNN.py

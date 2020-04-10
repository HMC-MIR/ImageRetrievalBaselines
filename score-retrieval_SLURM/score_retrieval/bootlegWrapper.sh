#!/bin/bash

#SBATCH -p RM-shared
#SBATCH -t 5:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --output=/pylon5/ir5phqp/dhyang/logs/multiprocess.out


source /home/dhyang/MIR5/bin/activate
python3 call_parallel.py "$1" "$2"

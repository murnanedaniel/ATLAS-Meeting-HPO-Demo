#!/bin/bash

#SBATCH -C gpu 
#SBATCH -t 4:00:00
#SBATCH -n 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -c 10
#SBATCH -q special
#SBATCH -o logs/%x-%j.out
#SBATCH -J ITk-sweep
#SBATCH -A m1759

conda activate exatrkx-test
export SLURM_CPU_BIND="cores"

echo -e "\nStarting sweeps\n"

for i in {0..7}; do
    echo "Launching task $i"
    srun --exact --gres=craynetwork:0 -u -N 1 -n 1 --ntasks-per-node=1 --gpus-per-task 1 wandb agent murnanedaniel/HPO_Demo/4h53uskw &
done
wait

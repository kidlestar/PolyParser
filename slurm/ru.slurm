#!/bin/bash
#SBATCH --job-name=BaseLine
##SBATCH --qos=qos_gpu-t4

###SBATCH --partition=gpu_p2

#SBATCH --gres=gpu:1
##SBATCH -C v100-16g
##SBATCH --cpus-per-task=5
#SBATCH --output=./zexp/BUD/ru/exp1/slurm.out
#SBATCH --error=./zexp/BUD/ru/exp1/slurm.err
##SBATCH --time=35:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

module purge
module load pytorch-gpu/py3/1.10.0

#srun python debug_linear.py -m Linear -g 0 -b 190 -r 0 -o 1 -e 5000 -l 2 -t /users/xudong.zhang/cdata/Utrain.txt -d /users/xudong.zhang/cdata/Udev.txt -u /users/xudong.zhang/cdata/Utest.txt -s ./LAYTest/model6/exp6/ -p ./LAYTest/model6/exp6/

srun python run.py train -c ./config/ru.cfg -f ./zexp/BUD/ru/exp1

#!/bin/bash

#SBATCH --job-name=my_experiment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/s/sutashu-tomonaga1/workspace/research-vrnn-pytorch/output/blizzard/output.log
#SBATCH --error=/home/s/sutashu-tomonaga1/workspace/research-vrnn-pytorch/output/error/blizzard/error.log

config_file=$1
python train.py --config-file $config_file

# use like `$ sbatch launch_experiment.sh config/experiment_1.txt`
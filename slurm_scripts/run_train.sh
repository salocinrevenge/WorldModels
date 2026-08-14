#!/bin/bash
#SBATCH --job-name=lewm_train
#SBATCH --chdir=/home/nicolas.silva/WorldModels/worldmodels/experiments/leWM/le-wm
#SBATCH --output=output_train.txt
#SBATCH --error=output_train.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Identifica a maquina
hostname 

# Ativa o ambiente virtual do uv
source .venv/bin/activate

# Executa o treino
python train.py +run_dir=./checkpoints
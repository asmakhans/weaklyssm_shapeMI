#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --time=1-00:00:00
#SBATCH --mem=70G

SCRIPT=$1
LABELNUM=$2
ROOT=$3

python "$SCRIPT" --labelnum "$LABELNUM" --gpu 0 --root_path "$ROOT"

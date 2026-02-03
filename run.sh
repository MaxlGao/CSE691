#!/bin/bash
#SBATCH -A grp_yanjiefu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00:00
#SBATCH -p general
#SBATCH -q public
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH -o output/OAS/vanila.%j.out
#SBATCH -e output/OAS/vanila.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akaush39@asu.edu

module load mamba/latest
eval "$(conda shell.bash hook)"
conda activate rlproject

cd /scratch/akaush39/CSE691/
python3 main_trimesh.py
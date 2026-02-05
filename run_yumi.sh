#!/bin/bash
#SBATCH -A grp_yanjiefu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 0-01:00:00
#SBATCH -p htc
#SBATCH -q public
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH -o output/Yumi/job.%j.out
#SBATCH -e output/Yumi/job.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akaush39@asu.edu

module load mamba/latest
eval "$(conda shell.bash hook)"
conda activate rlproject

cd /scratch/akaush39/CSE691/
xvfb-run -s "-screen 0 1024x768x24" python3 main_trimesh_yumi.py
#!/bin/bash

#SBATCH --job-name="bash"
#SBATCH --partition=v100-32g
#SBATCH --ntasks=4
#SBATCH --gres=gpu:2
#SBATCH --time=0-24:0
#SBATCH --chdir=.
#SBATCH --output=cout_training_HW2.txt
#SBATCH --error=cerr_training_HW2.txt


module load opt gcc python/3.9.13-gpu
sbatch_pre.sh
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml
sbatch_post.sh
#!/bin/bash

#SBATCH --job-name="bash"
#SBATCH --partition=v100-16g
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:0
#SBATCH --chdir=.
#SBATCH --output=cout_inference.txt
#SBATCH --error=cerr_inference.txt


module load opt gcc python/3.9.13-gpu
sbatch_pre.sh
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/qlora-out_final/ data/final/test_one.json prediction.json combine.json
sbatch_post.sh
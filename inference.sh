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
python3 -m ppl \
    --base_model_path Taiwan-LLM-7B-v2.0-chat/ \
    --peft_path axolotl/qlora-out_final/ \
    --test_data_path data/final/test_zero.json
python3 -m predict \
    --base_model_path Taiwan-LLM-7B-v2.0-chat/ \
    --peft_path axolotl/qlora-out_final/ \
    --test_file data/final/test_zero.json \
    --output_file prediction.json
python3 -m combine \
    --data_dir data/final/test_zero.json \
    --prediction_dir prediction.json \
    --output_dir combine.json
sbatch_post.sh
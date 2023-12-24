#!/bin/bash

#SBATCH --job-name="bash"
#SBATCH --partition=v100-32g
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:0
#SBATCH --chdir=.
#SBATCH --output=cout_inference.txt
#SBATCH --error=cerr_inference.txt


module load opt gcc python/3.9.13-gpu
sbatch_pre.sh
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/qlora-out_final_3000_one/ data/final/test_one.json prediction.json 3000_lora_1.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/qlora-out_final_3000_two/ data/final/test_two.json prediction.json 3000_lora_2.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/qlora-out_final_5000_zero/ data/final/test_zero.json prediction.json 5000_lora_0.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/qlora-out_final_5000_one/ data/final/test_one.json prediction.json 5000_lora_1.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/qlora-out_final_5000_two/ data/final/test_two.json prediction.json 5000_lora_2.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/qlora-out_final_10000_zero/ data/final/test_zero.json prediction.json 10000_lora_0.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/qlora-out_final_10000_one/ data/final/test_one.json prediction.json 10000_lora_1.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/qlora-out_final_10000_two/ data/final/test_two.json prediction.json 10000_lora_2.json
sbatch_post.sh
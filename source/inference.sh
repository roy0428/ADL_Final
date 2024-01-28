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
# inference with 3000 
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/results/qlora-out_final_3000_zero_2/ /home/p0428q/usr/ADL/HW3/data/final/test_zero.json prediction.json axolotl/output_json/3000_lora_0_2.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/results/qlora-out_final_3000_one_2/ /home/p0428q/usr/ADL/HW3/data/final/test_one.json prediction.json axolotl/output_json/3000_lora_1_2.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/results/qlora-out_final_3000_two_2/ /home/p0428q/usr/ADL/HW3/data/final/test_two.json prediction.json axolotl/output_json/3000_lora_2_2.json
# inference with 5000 
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/results/qlora-out_final_5000_zero_2/ /home/p0428q/usr/ADL/HW3/data/final/test_zero.json prediction.json axolotl/output_json/5000_lora_0_2.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/results/qlora-out_final_5000_one_2/ /home/p0428q/usr/ADL/HW3/data/final/test_one.json prediction.json axolotl/output_json/5000_lora_1_2.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/results/qlora-out_final_5000_two_2/ /home/p0428q/usr/ADL/HW3/data/final/test_two.json prediction.json axolotl/output_json/5000_lora_2_2.json
# inference with 10000 
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/results/qlora-out_final_10000_zero_2/ /home/p0428q/usr/ADL/HW3/data/final/test_zero.json prediction.json axolotl/output_json/10000_lora_0_2.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/results/qlora-out_final_10000_one_2/ /home/p0428q/usr/ADL/HW3/data/final/test_one.json prediction.json axolotl/output_json/10000_lora_1_2.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/results/qlora-out_final_10000_two_2/ /home/p0428q/usr/ADL/HW3/data/final/test_two.json prediction.json axolotl/output_json/10000_lora_2_2.json
# inference with 2300 
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/results/qlora-out_final_2300_zero/ /home/p0428q/usr/ADL/HW3/data/final/test_zero.json prediction.json axolotl/output_json/2300_lora_0.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/results/qlora-out_final_2300_one/ /home/p0428q/usr/ADL/HW3/data/final/test_one.json prediction.json axolotl/output_json/2300_lora_1.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/results/qlora-out_final_2300_two/ /home/p0428q/usr/ADL/HW3/data/final/test_two.json prediction.json axolotl/output_json/2300_lora_2.json
# inference with 5300 
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/results/qlora-out_final_5300_zero/ /home/p0428q/usr/ADL/HW3/data/final/test_zero.json prediction.json axolotl/output_json/5300_lora_0.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/results/qlora-out_final_5300_one/ /home/p0428q/usr/ADL/HW3/data/final/test_one.json prediction.json axolotl/output_json/5300_lora_1.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/results/qlora-out_final_5300_two/ /home/p0428q/usr/ADL/HW3/data/final/test_two.json prediction.json axolotl/output_json/5300_lora_2.json
# inference with 7300 
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/results/qlora-out_final_7300_zero/ /home/p0428q/usr/ADL/HW3/data/final/test_zero.json prediction.json axolotl/output_json/7300_lora_0.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/results/qlora-out_final_7300_one/ /home/p0428q/usr/ADL/HW3/data/final/test_one.json prediction.json axolotl/output_json/7300_lora_1.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/results/qlora-out_final_7300_two/ /home/p0428q/usr/ADL/HW3/data/final/test_two.json prediction.json axolotl/output_json/7300_lora_2.json
# inference with 12300 
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/results/qlora-out_final_12300_zero/ /home/p0428q/usr/ADL/HW3/data/final/test_zero.json prediction.json axolotl/output_json/12300_lora_0.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/results/qlora-out_final_12300_one/ /home/p0428q/usr/ADL/HW3/data/final/test_one.json prediction.json axolotl/output_json/12300_lora_1.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ axolotl/results/qlora-out_final_12300_two/ /home/p0428q/usr/ADL/HW3/data/final/test_two.json prediction.json axolotl/output_json/12300_lora_2.json
python3 -m eval \
    --data_dir output_json/ \
    --output_dir eval.json
sbatch_post.sh
#!/bin/bash

#SBATCH --job-name="training"
#SBATCH --partition=v100-32g
#SBATCH --ntasks=4
#SBATCH --gres=gpu:2
#SBATCH --time=2-0:00
#SBATCH --chdir=.
#SBATCH --output=cout_training_final.txt
#SBATCH --error=cerr_training_final.txt


module load opt gcc python/3.9.13-gpu
sbatch_pre.sh

pip3 install flash-attn==1.0.9

/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_3000.json" --output_dir="results/qlora-out_final_3000_zero_2"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_one_shot_3000.json" --output_dir="results/qlora-out_final_3000_one_2"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_two_shot_3000.json" --output_dir="results/qlora-out_final_3000_two_2"

/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_5000.json" --output_dir="results/qlora-out_final_5000_zero_2"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_one_shot_5000.json" --output_dir="results/qlora-out_final_5000_one_2"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_two_shot_5000.json" --output_dir="results/qlora-out_final_5000_two_2"

/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_10000.json" --output_dir="results/qlora-out_final_10000_zero_2"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_one_shot_10000.json" --output_dir="results/qlora-out_final_10000_one_2"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_two_shot_10000.json" --output_dir="results/qlora-out_final_10000_tw_2"

bash run.sh Taiwan-LLM-7B-v2.0-chat/ results/qlora-out_final_3000_zero_2/ /home/p0428q/usr/ADL/HW3/data/final/test_zero.json prediction.json output_json/3000_lora_0_2.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ results/qlora-out_final_3000_one_2/ /home/p0428q/usr/ADL/HW3/data/final/test_one.json prediction.json output_json/3000_lora_1_2.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ results/qlora-out_final_3000_two_2/ /home/p0428q/usr/ADL/HW3/data/final/test_two.json prediction.json output_json/3000_lora_2_2.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ results/qlora-out_final_5000_zero_2/ /home/p0428q/usr/ADL/HW3/data/final/test_zero.json prediction.json output_json/5000_lora_0_2.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ results/qlora-out_final_5000_one_2/ /home/p0428q/usr/ADL/HW3/ata/final/test_one.json prediction.json output_json/5000_lora_1_2.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ results/qlora-out_final_5000_two_2/ /home/p0428q/usr/ADL/HW3/data/final/test_two.json prediction.json output_json/5000_lora_2_2.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ results/qlora-out_final_10000_zero_2/ /home/p0428q/usr/ADL/HW3/data/final/test_zero.json prediction.json output_json/10000_lora_0_2.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ results/qlora-out_final_10000_one_2/ /home/p0428q/usr/ADL/HW3/data/final/test_one.json prediction.json output_json/10000_lora_1_2.json
bash run.sh Taiwan-LLM-7B-v2.0-chat/ results/qlora-out_final_10000_two_2/ /home/p0428q/usr/ADL/HW3/data/final/test_two.json prediction.json output_json/10000_lora_2_2.json

python3 -m eval \
    --data_dir output_json/ \
    --output_dir eval.json

sbatch_post.sh
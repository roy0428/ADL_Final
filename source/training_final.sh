#!/bin/bash

#SBATCH --job-name="training"
#SBATCH --partition=v100-32g
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --time=2-0:00
#SBATCH --chdir=.
#SBATCH --output=cout_training_final.txt
#SBATCH --error=cerr_training_final.txt


module load opt gcc python/3.9.13-gpu
sbatch_pre.sh
# training with 3000 
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_3000.json" --output_dir="results/qlora-out_final_3000_zero_2"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_one_shot_3000.json" --output_dir="results/qlora-out_final_3000_one_2"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_two_shot_3000.json" --output_dir="results/qlora-out_final_3000_two_2"
# training with 5000 
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_5000.json" --output_dir="results/qlora-out_final_5000_zero_2"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_one_shot_5000.json" --output_dir="results/qlora-out_final_5000_one_2"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_two_shot_5000.json" --output_dir="results/qlora-out_final_5000_two_2"
# training with 10000 
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_10000.json" --output_dir="results/qlora-out_final_10000_zero_2"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_one_shot_10000.json" --output_dir="results/qlora-out_final_10000_one_2"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_two_shot_10000.json" --output_dir="results/qlora-out_final_10000_two_2"
# training with all_real_2300 
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/all_real_2300.json" --output_dir="results/qlora-out_final_2300_zero"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_one_shot_all_real_2300.json" --output_dir="results/qlora-out_final_2300_one"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_two_shot_all_real_2300.json" --output_dir="results/qlora-out_final_2300_two"
# training with GPT3000+real2300
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_3000+2300.json" --output_dir="results/qlora-out_final_5300_zero"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_one_shot_3000+2300.json" --output_dir="results/qlora-out_final_5300_one"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_two_shot_3000+2300.json" --output_dir="results/qlora-out_final_5300_two"
# training with GPT5000+real2300
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_5000+2300.json" --output_dir="results/qlora-out_final_7300_zero"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_one_shot_5000+2300.json" --output_dir="results/qlora-out_final_7300_one"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_two_shot_5000+2300.json" --output_dir="results/qlora-out_final_7300_two"
# training with GPT10000+real2300
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_10000+2300.json" --output_dir="results/qlora-out_final_12300_zero"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_one_shot_10000+2300.json" --output_dir="results/qlora-out_final_12300_one"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_two_shot_10000+2300.json" --output_dir="results/qlora-out_final_12300_two"
sbatch_post.sh
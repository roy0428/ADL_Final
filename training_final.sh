#!/bin/bash

#SBATCH --job-name="bash"
#SBATCH --partition=v100-32g
#SBATCH --ntasks=4
#SBATCH --gres=gpu:2
#SBATCH --time=2-0:00
#SBATCH --chdir=.
#SBATCH --output=cout_training_final.txt
#SBATCH --error=cerr_training_final.txt


module load opt gcc python/3.9.13-gpu
sbatch_pre.sh

/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_one_shot_3000.json" --output_dir="./qlora-out_final_3000_one"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_two_shot_3000.json" --output_dir="./qlora-out_final_3000_two"

/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_5000.json" --output_dir="./qlora-out_final_5000_zero"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_one_shot_5000.json" --output_dir="./qlora-out_final_5000_one"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_two_shot_5000.json" --output_dir="./qlora-out_final_5000_two"

/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_10000.json" --output_dir="./qlora-out_final_10000_zero"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_one_shot_10000.json" --output_dir="./qlora-out_final_10000_one"
/home/p0428q/.local/bin/accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/home/p0428q/usr/ADL/HW3/data/final/train_two_shot_10000.json" --output_dir="./qlora-out_final_10000_two"
sbatch_post.sh
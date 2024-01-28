# Chinese Typo Correction with Taiwan-LLaMa

## Abstract
Developed a language model capable of identifying commonly misused words with an accuracy of 98.6%, surpassing the performance of GPT-4, which achieved only 88% accuracy.

## Data Generation and Preprocessing
Data Generation
```
python3 generator.py --number_of_data n --output_dir /path/to/output.json
```
Data Preprocessing
```
python3 preprocessing.py --data_dir /path/to/output.json --output_dir_0 /path/to/zero_shot.json --output_dir_1 /path/to/one_shot.json --output_dir_2 /path/to/two_shot.json
```
Do the following to process the training data
```
python3 generator.py \
    --number_of_data 1000 \
    --output_dir data/output.json

python3 preprocessing.py  \
    --data_dir data/output.json \
    --output_dir_0 data/train_1000_zero_shot.json \
    --output_dir_1 data/train_1000_one_shot.json \
    --output_dir_2 data/train_1000_two_shot.json 
```

## Training
Training
```
accelerate launch -m axolotl.cli.train examples/llama-2/qlora_final.yml --datasets.path="/path/to/dataset" --output_dir="/path/to/output/"
```
or modify the training_final.sh and do the following
```
bash training_final.sh
```
## Inference and Evaluation
Inference and Evaluation
```
bash run.sh /path/to/Taiwan-LLM-7B-v2.0-chat/ /path/to/qlora-out/ /path/to/test.json/ /path/to/prediction.json/ /path/to/combined_prediction.json/ 
```
or modify the inference.sh and do the following
```
bash inference.sh
```

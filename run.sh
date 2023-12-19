# ${1} Taiwan-LLM-7B-v2.0-chat/
# ${2} axolotl/qlora-out_final/
# ${3} data/final/test_zero.json
# ${4} prediction.json
# ${5} combine.json

python3 -m ppl \
    --base_model_path ${1} \
    --peft_path ${2} \
    --test_data_path ${3}
python3 -m predict \
    --base_model_path ${1} \
    --peft_path ${2} \
    --test_file ${3} \
    --output_file ${4}
python3 -m combine \
    --data_dir ${3} \
    --prediction_dir ${4} \
    --output_dir ${5}
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from peft import PeftModel
from utils import get_prompt, get_bnb_config
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9).",
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        required=True,
        help="Path to the saved PEFT checkpoint.",
    )
    parser.add_argument(
        "--test_file", type=str, default=None, required=True, help="Path to test data."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        required=True,
        help="Path to output data.",
    )
    args = parser.parse_args()

    # Load model
    bnb_config = get_bnb_config()

    if args.base_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    else:
        model_name = "yentinglin/Taiwan-LLM-7B-v2.0-chat"
        revision = "5073b2bbc1aa5519acdc865e99832857ef47f7c9"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load LoRA
    model = PeftModel.from_pretrained(model, args.peft_path)

    with open(args.test_file, "r") as f:
        data = json.load(f)

    # predict
    instructions = [get_prompt(x["instruction"]) for x in data["data"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    model.eval()
    keyword = "ASSISTANT: "  # split it to form the output format
    result_list = []
    cnt = 0
    for sentence in tqdm(instructions, desc="Generating text"):
        input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)
        output = model.generate(input_ids=input_ids, max_length=512)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        index = generated_text.find(keyword)
        if index != -1:
            prefix = generated_text[
                :index
            ].strip()  # prefix is the original instruction
            main_content = generated_text[index + len(keyword) :].strip()
        else:
            prefix = ""  # prefix is the original instruction
            main_content = generated_text.strip()

        result_dict = {
            # "id": data[cnt]["id"],
            "output": main_content,
        }
        result_list.append(result_dict)
        cnt = cnt + 1

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)

from argparse import ArgumentParser
import json


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="data/data.json", type=str)
    parser.add_argument("--output_dir_0", default="data/all_real_2300.json", type=str)
    parser.add_argument(
        "--output_dir_1", default="data/train_one_shot_all_real_2300.json", type=str
    )
    parser.add_argument(
        "--output_dir_2", default="data/train_two_shot_all_real_2300.json", type=str
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    with open(args.data_dir, "r") as json_file:
        data_list = json.load(json_file)

    result_0, result_1, result_2 = [], [], []
    for data in data_list:
        preprocessed_output, expect_answer = "", ""
        for word in data:
            if word == "在":
                preprocessed_output += "___"
                expect_answer += "在、"
            elif word == "再":
                preprocessed_output += "___"
                expect_answer += "再、"
            else:
                preprocessed_output += word

        instruction_0 = f"你是人工智慧助理，以下將提供被挖空的文本，你要對挖空的位置填入「在」或「再」，輸出格式為「答案：再、在」。以下為題目:{preprocessed_output} 答案："
        instruction_1 = f"你是人工智慧助理，以下將提供被挖空的文本，你要對挖空的位置填入「在」或「再」，輸出格式為「答案：再、在」。以下提供一個範例：題目：我___圖書館裡看書，等等要___回學校。答案：在、再。以下為題目:{preprocessed_output} 答案："
        instruction_2 = f"你是人工智慧助理，以下將提供被挖空的文本，你要對挖空的位置填入「在」或「再」，輸出格式為「答案：再、在」。以下提供兩個範例：題目：我___圖書館裡看書，等等要___回學校。答案：在、再。題目：我___過一下就要回家了，等等我___家你___打給我。答案：再、在、再。以下為題目:{preprocessed_output} 答案："
        result_0.append({"instruction": instruction_0, "output": expect_answer[:-1]})
        result_1.append({"instruction": instruction_1, "output": expect_answer[:-1]})
        result_2.append({"instruction": instruction_2, "output": expect_answer[:-1]})

    # data_json_0 = {"data": result_0}
    # data_json_1 = {"data": result_1}
    # data_json_2 = {"data": result_2}
    json.dump(result_0, open(args.output_dir_0, "w"), indent=2, ensure_ascii=False)
    json.dump(result_1, open(args.output_dir_1, "w"), indent=2, ensure_ascii=False)
    json.dump(result_2, open(args.output_dir_2, "w"), indent=2, ensure_ascii=False)

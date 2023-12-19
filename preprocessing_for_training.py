from argparse import ArgumentParser
import json


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir", default="data/test.json", type=str
    )
    parser.add_argument(
        "--output_dir", default="data/test_zero.json", type=str
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    with open(args.data_dir, "r") as json_file:
        data_list = json.load(json_file)

    result = []
    for data in data_list["data"]:
        data["instruction"] = f'你是人工智慧助理，以下將提供被挖空的文本，你要對挖空的位置填入「在」或「再」，輸出格式為「答案：再、在」。以下為題目:{data["instruction"]} 答案：'
        result.append(data)

    json.dump(result, open(args.output_dir, "w"), indent=2, ensure_ascii=False)

from argparse import ArgumentParser
import json


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="data/3000.json", type=str)
    parser.add_argument("--prediction_dir", default="data/prediction.json", type=str)
    parser.add_argument("--output_dir", default="combine.json", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    with open(args.data_dir, "r") as json_file:
        test_list = json.load(json_file)
    with open(args.prediction_dir, "r") as json_file:
        prediction_list = json.load(json_file)

    result = []
    for index in range(len(prediction_list)):
        data = {
            "instruction": test_list["data"][index]["instruction"],
            "output": test_list["data"][index]["output"],
            "prediction": prediction_list[index]["output"],
        }
        result.append(data)

    json.dump(result, open(args.output_dir, "w"), indent=2, ensure_ascii=False)

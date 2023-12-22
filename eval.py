import json
from tw_rouge import get_rouge
import os
from tqdm import tqdm
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir", default="all_results", type=str
    )
    parser.add_argument(
        "--output_dir", default="eval.json", type=str
    )
    args = parser.parse_args()
    return args


def get_accuracy(refs, preds):
    total_ans = 0
    correct_ans = 0
    for output, prediction in zip(refs, preds):
        for index in range(len(output)):
            if output[index] == ("在" or "再"):
                total_ans += 1
                if index >= len(prediction):
                    pass
                elif output[index] == prediction[index]:
                    correct_ans += 1
    accuracy = correct_ans / total_ans
    return accuracy

def main():
    args = parse_args()
    result = []
    directory = args.data_dir
    for filename in tqdm(sorted(os.listdir(directory))):
        refs, preds = [], []
        f = os.path.join(directory, filename)
        with open(f) as file:
            data_list = json.load(file)
            for data in data_list:
                refs.append(data["output"])
                if data["prediction"] != "":
                    preds.append(data["prediction"])
                else:
                    preds.append("無")
                
            ## Calculate rouge score
            rouge = get_rouge(preds, refs)
            
            ## Calculate accuracy
            accuracy_gpt = get_accuracy(refs[:100], preds[:100])
            accuracy_real = get_accuracy(refs[100:], preds[100:])
            accuracy_overall = (accuracy_gpt + accuracy_real) / 2
            result.append({"filename": filename, 
                           "rouge": rouge, 
                           "accuracy_gpt": accuracy_gpt, 
                           "accuracy_real": accuracy_real, 
                           "accuracy_overall": accuracy_overall})

    json.dump(result, open(args.output_dir, "w"), indent=2, ensure_ascii=False)
if __name__ == '__main__':
    main()

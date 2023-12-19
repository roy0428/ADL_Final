import json
from tw_rouge import get_rouge
import os
from tqdm import tqdm

def main():
    result = []
    directory = "all_results"
    for filename in tqdm(sorted(os.listdir(directory))):
        refs, preds = [], []
        total_ans = 0
        correct_ans = 0
        f = os.path.join(directory, filename)
        with open(f) as file:
            data_list = json.load(file)
            for data in data_list:
                refs.append(data["output"])
                preds.append(data["prediction"])
                
            ## Calculate rouge
            rouge = get_rouge(preds, refs)
            
            ## Calculate accuracy
            for output, prediction in zip(refs, preds):
                for index in range(len(output)):
                    if output[index] == ("在" or "再"):
                        total_ans += 1
                        if index >= len(prediction):
                            pass
                        elif output[index] == prediction[index]:
                            correct_ans += 1
            accuracy = correct_ans / total_ans
            result.append({"filename": filename, "rouge": rouge, "accuracy": accuracy})

    json.dump(result, open("eval.json", "w"), indent=2, ensure_ascii=False)
if __name__ == '__main__':
    main()

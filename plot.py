import matplotlib.pyplot as plt
import json
import ipdb

def _set_color(bars, left, right, color):
    for index in range(left, right):
        bars[index].set_color(f"{color}")
    return


def sort_key(list):
    custom_order = {'no_training_0_shot': 1, 
                    '3000_lora_2_2': 2, 
                    '5000_lora_1_2': 3, 
                    '10000_lora_0_2': 4,
                    'GPT3.5_0': 5,
                    'GPT4_0': 6,
                    '2300_lora_2': 7,
                    '5300_lora_2': 8,
                    '7300_lora_0': 9,
                    '12300_lora_0': 10,
                    # '7300_lora_1': 11,
                    # '7300_lora_2': 12,
                    # '12300_lora_0': 13,
                    # '12300_lora_1': 14,
                    # '12300_lora_2': 15,
                    # 'GPT4_0': 16,
                    # 'GPT4_1': 17,
                    # 'GPT4_2': 18
                    }
    return custom_order.get(list[0], float('inf'))

def main():
    with open("all_results/eval.json") as file:
        data_list = json.load(file)
    accuracy = []
    filename = []
    for data in data_list:
        accuracy.append(data["accuracy_overall"])
        filename.append(data["filename"][:-5])
    combine_list = list(zip(filename, accuracy))
    combine_list = sorted(combine_list, key=sort_key)
    accuracy = []
    filename = []
    for data in combine_list[:10]:
        filename.append(data[0])
        accuracy.append(data[1])
    # no_training = accuracy[-3:]
    # accuracy[-6:] = accuracy[9:15]
    # accuracy[9:12] = accuracy[:3]
    # accuracy[:3] = no_training

    bars = plt.bar(filename, accuracy)
    # plt.bar(filename, accuracy, color='blue')
    plt.xlabel('File Name', fontstyle='italic')
    plt.ylabel('Accuracy_overall')
    plt.xticks(rotation=45, ha='right')
    
    _set_color(bars, 8, 9, "lime") # no_training
    _set_color(bars, 6, 7, "limegreen")
    _set_color(bars, 9, 10, "greenyellow")
    # _set_color(bars, 9, 12, "black")
    # _set_color(bars, 12, 15, "purple")
    # _set_color(bars, 15, 18, "yellow")
    # _set_color(bars, 27, 30, "yellow")
    # _set_color(bars, 30, 36, "yellow")
    # _set_color(bars, 36, 39, "yellow") # no_training
    colors = {
        "1st": "lime",
        "2nd": "limegreen",
        "3rd": "greenyellow",
        # "2300+5000": "black",
        # "2300+10000": "purple",
        # "GPT4": "yellow",
    }
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)
    plt.show()


if __name__ == "__main__":
    main()

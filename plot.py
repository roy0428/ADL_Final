import matplotlib.pyplot as plt
import json

def _set_color(bars, left, right, color):
    for index in range(left, right):
        bars[index].set_color(f'{color}')
    return

def main():
    with open("all_results/eval.json") as file:
        data_list = json.load(file)
    accuracy = []
    for data in data_list:
        accuracy.append(data["accuracy_overall"])
        
    no_training = accuracy[-3:]
    accuracy[-6:] = accuracy[9:15]
    accuracy[9:12] = accuracy[:3]
    accuracy[:3] = no_training
    
    bars = plt.bar(range(len(data_list)), list(accuracy))
    _set_color(bars, 0, 3, "green")
    _set_color(bars, 3, 6, "blue")
    _set_color(bars, 6, 9, "red")
    _set_color(bars, 9, 12, "black")
    _set_color(bars, 12, 15, "purple")
    _set_color(bars, 15, 18, "yellow")
    colors = {'no_training':'green', '3000':'blue', '5000':'red', '10000':'black', 'GPT3.5':'purple', 'GPT4':'yellow'}         
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)
    plt.show()

if __name__ == "__main__":
    main()
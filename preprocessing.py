import json

if __name__ == "__main__":

    with open("data/output.json", "r") as json_file:
        data_list = json.load(json_file)

    result = []
    for data in data_list:
        preprocessed_output = ""
        expect_answer = ""

        for word in data:
            if word == "在":
                preprocessed_output += "___"
                expect_answer += "在、"
            elif word == "再":
                preprocessed_output += "___"
                expect_answer += "再、"
            else:
                preprocessed_output += word
        
        result.append({"instruction": preprocessed_output, "output": expect_answer[:-1]})

    data_json = {"data": result}
    json.dump(data_json, open("data/test.json", "w"), indent=2, ensure_ascii=False)

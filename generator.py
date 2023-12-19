from openai import OpenAI
from tqdm import tqdm
import json

def generator():
    api_key = "APIKEY"
    client = OpenAI(api_key=api_key)
    system_content = "你的任務是以提供的文字造句，句子長度不超過五十個字。"
    user_content = "請造一完整且有趣的句子，內容須包含「在」及「再」，且「在」不要在句首"
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        max_tokens=200,
    )
    output = response.choices[0].message.content
    print(output)
    return output


def main():
    data = []
    for i in tqdm(range(100)):
        data.append(generator())
        
    data_json = data
    json.dump(data_json, open("data/output.json", "w"), indent=2, ensure_ascii=False)
    return


if __name__ == "__main__":
    main()

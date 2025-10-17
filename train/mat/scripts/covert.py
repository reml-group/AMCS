import json
import re

def convert_dataset(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    converted_data = []
    for item in original_data:
        match = re.search(r'The answer is: (.*)', item['response'])
        final_answer = match.group(1).strip() if match else "CANNOT_EXTRACT"

        converted_item = {
            "problem": item["query"],
            "final_answer": final_answer
        }
        converted_data.append(converted_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item) + '\n')

input_file = "path"
output_file = "path"
convert_dataset(input_file, output_file)


print(f"转换完成！新文件已保存至: {output_file}")

import json
import os
import random
from typing import List
import math

def sample_questions(input_filepath: str, output_filepath: str, num_samples: int):
    with open(input_filepath, 'r') as f:
        questions = json.load(f)

    sampled_questions = random.sample(questions, min(num_samples, len(questions)))

    with open(output_filepath, 'w') as f:
        json.dump(sampled_questions, f, indent=4)

    print(f"Saved {len(sampled_questions)} sampled questions to {output_filepath}")


def split_questions(input_filepath: str, output_dir: str, questions_per_file: int):
    with open(input_filepath, 'r') as f:
        questions = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for i in range(0, len(questions), questions_per_file):
        chunk = questions[i:i + questions_per_file]
        output_filepath = os.path.join(output_dir, f"questions_part_{i // questions_per_file + 1}.json")
        with open(output_filepath, 'w') as f:
            json.dump(chunk, f, indent=4)

        print(f"Saved {len(chunk)} questions to {output_filepath}")


def split_questions_uniformly(input_filepath: str, output_directory: str, num_files: int):

    with open(input_filepath, 'r') as f:
        questions = json.load(f)

    total_questions = len(questions)
    questions_per_file = math.ceil(total_questions / num_files)

    os.makedirs(output_directory, exist_ok=True)

    for i in range(num_files):
        start_idx = i * questions_per_file
        end_idx = min(start_idx + questions_per_file, total_questions)
        questions_subset = questions[start_idx:end_idx]

        output_filepath = os.path.join(output_directory, f"questions_part_{i + 1}.json")
        with open(output_filepath, 'w') as f_out:
            json.dump(questions_subset, f_out, indent=4)

        print(f"Saved {len(questions_subset)} questions to {output_filepath}")



if __name__ == "__main__":
    split_questions_uniformly("path", "output_directory", 2)


import argparse
from transformers import AutoTokenizer
import json

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=True, help="Model ID")
parser.add_argument("--token", type=str, required=True, help="Access token")
parser.add_argument("--file_path", type=str, required=True, help="Path to the file")

args = parser.parse_args()

model_id = args.model

print(f"Model ID: {model_id}")
print(f"File path: {args.file_path}")

tokenizer = AutoTokenizer.from_pretrained(model_id, token=args.token)

# Open json file
with open(args.file_path, "r") as file:
    data = json.load(file)

# Print datapoints
print(f"Number of datapoints in {args.file_path}: {len(data)}")

count_over_256 = 0
# Iterate through datapoints and count token length of answer-llm
for i, datapoint in enumerate(data):
    answer_llm = datapoint.get("answer-llm", None)
    # print(f"Datapoint {i}: {answer_llm}")
    if answer_llm:
        token_length = len(tokenizer.encode(answer_llm[0]))
        # print(f"Token length of answer-llm: {token_length}")
        if token_length >= 255:
            count_over_256 += 1
    else:
        print("No answer-llm found")

print(f"Number of datapoints with token length of 256: {count_over_256}")

# Example usage:
# python count_token_length.py --model "meta-llama/Meta-Llama-3.1-8B-Instruct" --token "HFTOKEN" --file_path "Path/to/file.json"
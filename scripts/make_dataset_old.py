# Import json file
import json
import sys
import os
import transformers
import torch
import time  # Import the time module
from tqdm import tqdm
import numpy as np
import random
from transformers import pipeline, set_seed

# Function to set all random seeds
def set_all_seeds(seed):
    # Set seed for Python's built-in random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch (both CPU and GPU)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set seed in the transformers library
    set_seed(seed)

# Example usage
seed_value = 42
set_all_seeds(seed_value)

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_id ="google/gemma-2-9b-it",

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Constants for file paths
TRAIN_FILE = '../data/train.json'
VALID_FILE = '../data/valid.json'
TEST_FILE = '../data/test.json'

def check_file_exists(filepath):
    """Check if the file exists."""
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist.")
        sys.exit(1)

def load_json_file(filepath):
    """Load a JSON file and return the data."""
    check_file_exists(filepath)
    with open(filepath, "r") as file:
        return json.load(file)

def generate_llm_responses(datapoint, pipeline):
    """Generate LLM responses for a given datapoint using a pipeline."""
    question = datapoint['question']

    # Generate the answer
    messages = [{"role": "user", "content": question}]
    outputs = pipeline(messages, max_new_tokens=200)
    print(f"Type outputs {type(outputs)}")
    print(f"Type outputs[0] {type(outputs[0])}")
    print(outputs[0])
    answer = outputs[0]["generated_text"][-1]['content']

    # Generate the follow-up question
    messages = [{"role": "user", "content": f"Give one follow-up question for: {answer}"}]
    outputs = pipeline(messages, max_new_tokens=64)
    followup = outputs[0]["generated_text"][-1]['content']

    # Add the generated answer and follow-up to the datapoint
    datapoint['answer-llm'] = answer
    datapoint['follow-up-llm'] = followup

    return datapoint

def process_dataset(filepath, pipeline, output_suffix="-llm"):
    """Process a dataset by generating LLM responses and saving the updated data."""
    # Load the dataset
    data = load_json_file(filepath)

    # Process each datapoint in the dataset
    for datapoint in tqdm(data):
        datapoint = generate_llm_responses(datapoint, pipeline)
    
    # Save the modified dataset
    output_filepath = filepath.replace('.json', f'{output_suffix}.json')
    save_json_file(data, output_filepath)

def save_json_file(data, filepath):
    """Save data to a JSON file."""
    with open(filepath, "w") as file:
        json.dump(data, file, indent=4)

def main(pipeline):
    """Main function to process all datasets."""
    process_dataset(VALID_FILE, pipeline)
    # process_dataset(TRAIN_FILE, pipeline)
    # process_dataset(TEST_FILE, pipeline)

# Example usage:
# Assuming you have a defined pipeline, you can call the main function:
# main(pipeline)
if __name__ == "__main__":
    main(pipeline)  # Replace `pipeline` with your actual pipeline instance

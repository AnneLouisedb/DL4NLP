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
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, GemmaForCausalLM
import argparse
import torch.nn.functional as F
import h5py

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
    question = f"{datapoint['question']} Keep your answer short."

    inputs = tokenizer([question], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, return_dict_in_generate=True, output_logits=True, max_new_tokens=256)

    inputs = inputs.to('cpu')
    logits_answer = torch.stack(outputs.logits).squeeze(1)
    sequences = outputs.sequences

    answer = tokenizer.batch_decode(
        sequences[:, inputs.input_ids.shape[1]:], 
        skip_special_tokens=True,
    )

    follow_up_prompt = f"Give one follow-up question for: {answer}"
    inputs_follow_up = tokenizer([follow_up_prompt], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs_follow_up = model.generate(**inputs_follow_up, return_dict_in_generate=True, output_logits=True, max_new_tokens=64)
        
    inputs_follow_up = inputs_follow_up.to('cpu')
    logits_follow_up = torch.stack(outputs_follow_up.logits).squeeze(1)
    sequence_follow_up = outputs_follow_up.sequences
    follow_up = tokenizer.batch_decode(
        sequence_follow_up[:, inputs_follow_up.input_ids.shape[1]:], 
        skip_special_tokens=True
    )

    datapoint['answer-llm'] = answer
    datapoint['follow-up-llm'] = follow_up

    return datapoint, logits_answer, logits_follow_up

# New function to append tensors to an HDF5 file
def save_tensors_iteratively(tensor, hdf5_path, dataset_name):
    """Save tensor iteratively in an HDF5 file."""
    with h5py.File(hdf5_path, 'a') as f:
        if dataset_name in f:
            # If the dataset already exists, resize it and append the new tensor
            existing_data = f[dataset_name]
            new_size = existing_data.shape[0] + tensor.shape[0]
            existing_data.resize((new_size,) + existing_data.shape[1:])
            existing_data[-tensor.shape[0]:] = tensor
        else:
            # Create the dataset if it doesn't exist
            maxshape = (None,) + tensor.shape[1:]  # Allow unlimited first dimension
            tensor = tensor.cpu().numpy()
            f.create_dataset(dataset_name, data=tensor, maxshape=maxshape)

# Modified process_dataset function
def process_dataset(filepath, pipeline, output_suffix="-llm"):
    """Process a dataset by generating LLM responses and saving the updated data iteratively."""
    # Load the dataset
    data = load_json_file(filepath)

    hdf5_answer_path = filepath.replace('data/', 'data/llama/').replace('.json', '-answer.h5')
    hdf5_follow_up_path = filepath.replace('data/', 'data/llama/').replace('.json', '-follow-up.h5')

    # Process each datapoint in the dataset
    for datapoint in tqdm(data):
        datapoint, logits_ans, logits_follow_up = generate_llm_responses(datapoint, pipeline)

        # Convert logits to CPU and half precision
        logits_ans = logits_ans.to('cpu').to(torch.float16)
        logits_follow_up = logits_follow_up.to('cpu').to(torch.float16)

        # Save logits iteratively
        save_tensors_iteratively(logits_ans, hdf5_answer_path, 'logits_answers')
        save_tensors_iteratively(logits_follow_up, hdf5_follow_up_path, 'logits_follow_ups')

        # Empty the cache
        torch.cuda.empty_cache()

    print(f"Tensors saved iteratively in {hdf5_answer_path} and {hdf5_follow_up_path}")

def save_json_file(data, filepath):
    """Save data to a JSON file."""
    with open(filepath, "w") as file:
        json.dump(data, file, indent=4)

def main(pipeline):
    """Main function to process all datasets."""
    # Create the parser

    # process_dataset(VALID_FILE, pipeline)
    process_dataset(TRAIN_FILE, pipeline)
    # process_dataset(TEST_FILE, pipeline)

# Example usage:
# Assuming you have a defined pipeline, you can call the main function:
# main(pipeline)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description here")

    # Add arguments
    parser.add_argument('--model', default='llama', type=str, required=True, help='Path to the input file')

    # Parse the arguments
    args = parser.parse_args()

    # Example usage
    seed_value = 42
    set_all_seeds(seed_value)

    device = "cuda"
    access_token="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"

    if args.model == "llama":
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
        model = LlamaForCausalLM.from_pretrained(model_id, token=access_token)
        print("==================================")
        print(f"Model and tokenizer for {args.model} initialized")
        print("==================================")

    elif args.model == "gemma":
        model_id ="google/gemma-2-9b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
        model = GemmaForCausalLM.from_pretrained(model_id, token=access_token)
        print("==================================")
        print(f"Model and tokenizer for {args.model} initialized")
        print("==================================")

    model = model.to(device)
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    main(pipeline)  # Replace `pipeline` with your actual pipeline instance

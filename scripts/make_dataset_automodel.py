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
from transformers import LlamaForCausalLM, AutoTokenizer, Gemma2ForCausalLM
import argparse
import torch.nn.functional as F
from safetensors.torch import save_file

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

# Modified process_dataset function
def process_dataset(filepath, pipeline, output_suffix="-llm"):
    """Process a dataset by generating LLM responses and saving the updated data iteratively."""
    # Load the dataset
    data = load_json_file(filepath)
    print(len(data))
    # Make tensor of len data and 2 columns
    tensor_ans = torch.zeros(len(data), 2, 256)
    tensor_follow_up = torch.zeros(len(data), 2, 64)
    # Process each datapoint in the dataset
    for i, datapoint in tqdm(enumerate(data)):
        datapoint, logits_ans, logits_follow_up = generate_llm_responses(datapoint, pipeline)

        path = filepath.replace('data/', f'data/{args.model}/')
        save_json_file(data, path)

        # Convert logits to CPU and half precision
        logits_ans = logits_ans.cpu().to(torch.float16)
        print(f"Shape logits_ans before padding {logits_ans.shape}")
        # Add padding to the logits_ans tensor
        # if args.model == "llama":
        #     if logits_ans.shape[1] < 256:
        #         padding = torch.zeros(logits_ans.shape[0], 256 - logits_ans.shape[1], logits_ans.shape[2])
        #         logits_ans = torch.cat((logits_ans, padding), dim=1)

        # elif args.model == "gemma":
        if logits_ans.shape[0] < 256:
            padding = torch.zeros(256 - logits_ans.shape[0], logits_ans.shape[1])
            logits_ans = torch.cat((logits_ans, padding), dim=0)

            
        print(f"Shape logits_ans after padding {logits_ans.shape}")
        max_val, idx = torch.max(logits_ans, dim=-1)

        print(f"Shape idx : {idx.shape}")
        print(f"Shape max_val : {max_val.shape}")

        tensor_ans[i, 0, :] = idx
        
        tensor_ans[i, 1, :] = max_val


        logits_follow_up = logits_follow_up.cpu().to(torch.float16)
        # print(f"Logits follow up shape : {logits_follow_up.shape}")
        # Add padding to the logits_follow_up tensor
        print(f"Shape logits_follow_up before padding {logits_follow_up.shape}")
        # if args.model == "llama":
        #     if logits_follow_up.shape[1] < 64:
        #         padding = torch.zeros(logits_follow_up.shape[0], 64 - logits_follow_up.shape[1], logits_follow_up.shape[2])
        #         logits_follow_up = torch.cat((logits_follow_up, padding), dim=1)

        # elif args.model == "gemma":
        if logits_follow_up.shape[0] < 64:
            padding = torch.zeros(64 - logits_follow_up.shape[0], logits_follow_up.shape[1])
            logits_follow_up = torch.cat((logits_follow_up, padding), dim=0)

        print(f"Shape logits_follow_up after padding {logits_follow_up.shape}")
        max_val, idx = torch.max(logits_follow_up, dim=-1)
        print(f"Shape idx : {idx.shape}")
        print(f"Shape max_val : {max_val.shape}")
        # print(f"Shape max _val {max_val.shape}")
        # print(f"Shape idx {idx.shape}")
        tensor_follow_up[i, 0, :] = idx
        tensor_follow_up[i, 1, :] = max_val
        
        # Empty the cache
        torch.cuda.empty_cache()

        # If i is a multiple of 100, save the tensors to disk
        if i % 100 == 0:
            path = filepath.replace('data/', f'data/{args.model}/tensors/').replace('.json', f'-ans.pt')
            torch.save(tensor_ans, path)
            path = filepath.replace('data/', f'data/{args.model}/tensors/').replace('.json', f'-follow-up.pt')
            torch.save(tensor_follow_up, path)

    path = filepath.replace('data/', f'data/{args.model}/tensors/').replace('.json', f'-ans.pt')
    torch.save(tensor_ans, path)
    path = filepath.replace('data/', f'data/{args.model}/tensors/').replace('.json', f'-follow-up.pt')
    torch.save(tensor_follow_up, path)

def save_json_file(data, filepath):
    """Save data to a JSON file."""
    with open(filepath, "w") as file:
        json.dump(data, file, indent=4)

def main(pipeline):
    """Main function to process all datasets."""
    # Create the parser

    # process_dataset(VALID_FILE, pipeline)
    # process_dataset(TRAIN_FILE, pipeline)
    process_dataset(TEST_FILE, pipeline)

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


    elif args.model == "gemma":
        model_id ="google/gemma-2-9b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
        model = Gemma2ForCausalLM.from_pretrained(model_id, token=access_token)

    print("==================================")
    print(f"Model and tokenizer for {args.model} initialized")
    print("==================================")

    model = model.to(device)
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    main(pipeline)  # Replace `pipeline` with your actual pipeline instance

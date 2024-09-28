# Import necessary libraries
import json
import sys
import os
import torch
from tqdm import tqdm
import numpy as np
import random
from transformers import set_seed, AutoTokenizer
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to set all random seeds
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

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

def generate_llm_responses(datapoint, model, tokenizer):
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

def process_dataset(model, tokenizer):
    """Process a dataset by generating LLM responses and saving the updated data iteratively."""
    file_path = args.folder_path + f"/{args.split}.json"
    save_path = f"{args.folder_path}/{args.model.split('/')[-1]}/"
    tensor_path = f"{save_path}/tensors/"

    data = load_json_file(file_path)

    # Print the number of datapoints in the dataset
    print(f"Number of datapoints in {file_path}: {len(data)}")

    print(f"Generating LLM responses for {args.split} split...")
    tensor_ans = torch.zeros(len(data), 2, 256)
    tensor_follow_up = torch.zeros(len(data), 2, 64)

    for i, datapoint in tqdm(enumerate(data)):
        datapoint, logits_ans, logits_follow_up = generate_llm_responses(datapoint, model, tokenizer)

        save_json_file(data, save_path + args.split + '.json')

        logits_ans = logits_ans.cpu().to(torch.float16)
        if logits_ans.shape[0] < 256:
            padding = torch.zeros(256 - logits_ans.shape[0], logits_ans.shape[1])
            logits_ans = torch.cat((logits_ans, padding), dim=0)

        max_val, idx = torch.max(logits_ans, dim=-1)
        tensor_ans[i, 0, :] = idx
        tensor_ans[i, 1, :] = max_val

        logits_follow_up = logits_follow_up.cpu().to(torch.float16)
        if logits_follow_up.shape[0] < 64:
            padding = torch.zeros(64 - logits_follow_up.shape[0], logits_follow_up.shape[1])
            logits_follow_up = torch.cat((logits_follow_up, padding), dim=0)

        max_val, idx = torch.max(logits_follow_up, dim=-1)
        tensor_follow_up[i, 0, :] = idx
        tensor_follow_up[i, 1, :] = max_val

        torch.cuda.empty_cache()

        if i % 100 == 0:
            path = tensor_path + f"{args.split}_ans.pt"
            torch.save(tensor_ans, path)
            path = tensor_path + f"{args.split}_follow-up.pt"
            torch.save(tensor_follow_up, path)

    path = tensor_path + f"{args.split}_ans.pt"
    torch.save(tensor_ans, path)
    path = tensor_path + f"{args.split}_follow-up.pt"
    torch.save(tensor_follow_up, path)

def save_json_file(data, filepath):
    """Save data to a JSON file."""
    with open(filepath, "w") as file:
        json.dump(data, file, indent=4)

def main(args):
    """Main function to process the dataset."""
    set_all_seeds(args.seed)
    
    model_id = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=args.token)

    if "gemma" in model_id:
        from transformers import Gemma2ForCausalLM
        model = Gemma2ForCausalLM.from_pretrained(model_id, token=args.token)

    elif "meta-llama" in model_id:
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(model_id, token=args.token)

    
    # Make folder for model output
    os.makedirs(f"{args.folder_path}/{model_id.split('/')[-1]}/tensors/", exist_ok=True)

    model = model.to(device)
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    process_dataset(model, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for generating LLM responses")
    parser.add_argument('--model', type=str, required=True, help='Model to use ["google/gemma-2-9b-it", "meta-llama/Meta-Llama-3.1-8B-Instruct"]', default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument('--split', type=str, required=True, help='Split to process ["train", "valid", "test"]')
    parser.add_argument('--folder_path', type=str, required=True, help='Path to data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--token', type=str, required=True, help='Hugging Face API token')

    args = parser.parse_args()

    main(args)

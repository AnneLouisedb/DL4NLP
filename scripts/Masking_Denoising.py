import random
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch


def mask_tokens(text, alpha, tokenizer, mask_token="[MASK]"):
    """
    Randomly mask tokens in the input text.
    
    :param text: Input text to mask
    :param alpha: Masking ratio (0 to 1), denoting the percentage of tokens we want to mask.
    :param tokenizer: Tokenizer to use for tokenization
    :param mask_token: Token to use for masking (default: "[MASK]")
    :return: Masked text and indices of masked tokens
    """
    # Tokenize the input text
    tokens = tokenizer.tokenize(text)
    
    # Calculate the number of tokens to mask
    num_to_mask = int(len(tokens) * alpha)
    
    # Randomly select indices to mask
    mask_indices = random.sample(range(len(tokens)), num_to_mask)
    
    # Apply masking
    masked_tokens = tokens.copy()
    for idx in mask_indices:
        masked_tokens[idx] = mask_token
    
    # Convert back to text
    masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
    
    return masked_text, mask_indices

def fill_masked_tokens(masked_text, model, tokenizer):
    """
    Fill masked tokens in the input text using an LLM.
    
    :param masked_text: Text with masked tokens (using <mask> or [MASK])
    :param model: Language model to use for filling masked tokens
    :param tokenizer: Tokenizer associated with the model
    :return: Text with filled masked tokens
    """
    # Determine the mask token
    mask_token = tokenizer.mask_token
    
    
    # Encode the masked text
    inputs = tokenizer(masked_text, return_tensors="pt").to(device)
    masked_token_index = torch.where(inputs["input_ids"] == tokenizer.masked_token_id)[1]
    # generate predictions
    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits
    
    #decode
    for i, mask_index in enumerat(masked_token_index):
        mask_token_logits = pred[0, mask_index, :]
        top_tokens = torch.topk(mask_token_logits, k=5, dim=0).indices.tolist()
        for token in top_tokens:
            word = tokenizer.decode([token])
            print(word)
    
    return masked_text

# def process_data_with_masking(model_name, split,  model, tokenizer, alphas = [0.01, 0.02, 0.15, 0.5, 0.9], num_iterations = 10):
def process_data_with_masking(model_name, split,  model, tokenizer, alphas = [0.5], num_iterations = 3):
    """
    Process data from a JSON file, apply masking and denoising to specific fields,
    and save the denoised results back to the dictionary.
    
    :param model_name: Name of the model folder
    :param split: Split of the data (e.g., 'train', 'valid', 'test')
    :param model: Language model to use for filling masked tokens
    :param tokenizer: Tokenizer associated with the model
    :param alphas: List of masking ratios to apply
    :param num_iterations: Number of times to perform masking and denoising (default: 10)
    """
    # Construct the file path
    file_path = f'../data/{model_name}/{split}.json'
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    # Load the JSON data
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Process each item in the data
    for item in tqdm(data, desc=f"Processing {split} data"):
        for key in ['answer-llm', 'follow-up-llm']:
            if key in item:
                original_text = item[key]
                for alpha in alphas:
                    for i in range(1, num_iterations + 1):
                        # Apply masking
                        masked_text, _ = mask_tokens(item[key], alpha, tokenizer)
                        
                        # Apply denoising
                        denoised_text = fill_masked_tokens(masked_text, model, tokenizer)
                        
                        # Save the result back to the dictionary
                        item[f"{key}_alpha_{alpha}_{i}"] = denoised_text
                        print(denoised_text)
    
    # Save the updated data back to the file
    output_file_path = f'data/{model_name}/{split}.json'
    with open(output_file_path, 'w') as file:
        json.dump(data, file, indent=4)
    
    print(f"Processed data saved to: {output_file_path}")

# Example:

model_name = "llama"
split = "valid"  # or "train" or "test"
alpha = 0.15  # 15% masking ratio
device = "cuda"
# Assuming you have already loaded your model and tokenizer

# Model ID from Hugging Face Hub
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
access_token="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"

# Load tokenizer and model from Hugging Face Hub (requires access token)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
model = AutoModelForMaskedLM.from_pretrained(model_id, token=access_token)
model = model.to("cuda")

process_data_with_masking(model_name, split, model, tokenizer)

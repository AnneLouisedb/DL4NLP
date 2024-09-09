import random
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


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
    
    :param masked_text: Text with masked tokens
    :param model: Language model to use for filling masked tokens
    :param tokenizer: Tokenizer associated with the model
    :return: Text with filled masked tokens
    """
    # Encode the masked text
    inputs = tokenizer(masked_text, return_tensors="pt")
    
    # Generate output from the model
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    
    # Decode the output
    filled_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return filled_text

def process_data_with_masking(model_name, split, alpha, model, tokenizer):
    """
    Process data from a JSON file, apply masking and denoising to specific fields,
    and save the denoised results back to the dictionary.
    
    :param model_name: Name of the model folder
    :param split: Split of the data (e.g., 'train', 'valid', 'test')
    :param alpha: Masking ratio
    :param model: Language model to use for filling masked tokens
    :param tokenizer: Tokenizer associated with the model
    """
    # Construct the file path
    file_path = f'data/{model_name}/{split}.json'
    
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
                # Apply masking
                masked_text, _ = mask_tokens(item[key], alpha, tokenizer)
                
                # Apply denoising
                denoised_text = fill_masked_tokens(masked_text, model, tokenizer)
                
                # Save the result back to the dictionary
                item[f"{key}_alpha_{alpha}"] = denoised_text
    
    # Save the updated data back to the file
    output_file_path = f'data/{model_name}/{split}.json'
    with open(output_file_path, 'w') as file:
        json.dump(data, file, indent=4)
    
    print(f"Processed data saved to: {output_file_path}")

# Example:

model_name = "llama"
split = "valid"  # or "train" or "test"
alpha = 0.15  # 15% masking ratio

# Assuming you have already loaded your model and tokenizer
# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
# model = AutoModelForCausalLM.from_pretrained(model_id, token=access_token)
# model = model.to("cuda")

process_data_with_masking(model_name, split, alpha, model, tokenizer)

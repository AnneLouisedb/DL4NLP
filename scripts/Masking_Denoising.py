import random
import json
import os
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
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
    tokens = tokenizer.tokenize(text)
    num_to_mask = int(len(tokens) * alpha)
    mask_indices = random.sample(range(len(tokens)), num_to_mask)
    
    masked_tokens = tokens.copy()
    for idx in mask_indices:
        masked_tokens[idx] = mask_token
    
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
    inputs = tokenizer(masked_text, return_tensors="pt").to(device)
    masked_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    filled_tokens = inputs["input_ids"].clone()

    for i, mask_index in enumerate(masked_token_index):
        mask_token_logits = logits[0, mask_index, :]
        predicted_token_id = torch.argmax(mask_token_logits, dim=-1)
        filled_tokens[0, mask_index] = predicted_token_id
    
    filled_text = tokenizer.decode(filled_tokens[0], skip_special_tokens=True)
    
    return filled_text

def process_data_with_masking(model_name, split, model, tokenizer, alphas=[0.5], num_iterations=3):
    """
    Process data from a JSON file, apply masking and denoising to specific fields,
    and save the denoised results back to the dictionary.
    
    :param model_name: Name of the model folder
    :param split: Split of the data (e.g., 'train', 'valid', 'test')
    :param model: Language model to use for filling masked tokens
    :param tokenizer: Tokenizer associated with the model
    :param alphas: List of masking ratios to apply
    :param num_iterations: Number of times to perform masking and denoising (default: 3)
    """
    file_path = f'../data/{model_name}/{split}.json'
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    for item in tqdm(data, desc=f"Processing {split} data"):
        for key in ['answer-llm', 'follow-up-llm']:
            if key in item:
                original_text = item[key]
                for alpha in alphas:
                    for i in range(1, num_iterations + 1):
                        masked_text, _ = mask_tokens(original_text, alpha, tokenizer)
                        denoised_text = fill_masked_tokens(masked_text, model, tokenizer)
                        
                        item[f"{key}_alpha_{alpha}_{i}_denoised"] = denoised_text
                        print(f"Denoised Text: {denoised_text}")
    
    output_file_path = f'../data/{model_name}/{split}_denoised.json'
    with open(output_file_path, 'w') as file:
        json.dump(data, file, indent=4)
    
    print(f"Processed data saved to: {output_file_path}")

# Example:

model_name = "llama"
split = "valid"
device = "cuda"

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
access_token = "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"


tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
model = LlamaForCausalLM.from_pretrained(model_id, token=access_token)
model = model.to(device)

process_data_with_masking(model_name, split, model, tokenizer)


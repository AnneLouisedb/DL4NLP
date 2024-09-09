import random
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

# Example usage:
alpha = 0.15  # 15% masking ratio

# Assuming you have already loaded your model and tokenizer
# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
# model = AutoModelForCausalLM.from_pretrained(model_id, token=access_token)
# model = model.to("cuda")

sample_text = "The quick brown fox jumps over the lazy dog."

# Mask tokens
masked_text, mask_indices = mask_tokens(sample_text, alpha, tokenizer)
print(f"Masked text: {masked_text}")

# Fill masked tokens
filled_text = fill_masked_tokens(masked_text, model, tokenizer)
print(f"Filled text: {filled_text}")

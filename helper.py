import numpy as np
from tqdm import tqdm
from llama_cpp import Llama
import re
from typing import List, Dict
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load models to fill the masked tokens
model_name = "t5-large"
T5tokenizer_large = T5Tokenizer.from_pretrained(model_name)
T5model_large = T5ForConditionalGeneration.from_pretrained(model_name)

model_name = "t5-small"
T5tokenizer_small = T5Tokenizer.from_pretrained(model_name)
T5model_small = T5ForConditionalGeneration.from_pretrained(model_name)


############################### HELPER FUNCTIONS ####################
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def get_ll(llm: Llama, text: str):
    """Calculate the log-likelihood of the given text using the Llama model.
       If a runtime error occurs during evaluation, return None or an alternative value."""
    try:
        tokenized_text = llm.tokenizer().encode(text)
        llm.reset()
        llm.eval(tokenized_text)  # runtime error might occur here
        logits = np.array(llm._scores)
        softmax_logits = softmax(logits)
        log_likelihood = 0.0
        for i, token_id in enumerate(tokenized_text):
            prob = softmax_logits[i, token_id]
            log_likelihood += np.log(prob)

        return log_likelihood

    except RuntimeError as e:
        # Handle the RuntimeError (e.g., return None or log the error)
        print(f"RuntimeError occurred: {e}")
        return None  # You can return a default value or some other indicator



def get_lls(llm: Llama, texts: [str], disable_tqdm):

    assert isinstance(texts, list), "texts must be a list"
    
    lls = []
    for text in tqdm(
        texts, desc="Log Likelihood for Text Estimation", disable=disable_tqdm

    ):  
        assert isinstance(text[0], str), f"texts must be a string {text[0]}"
        lls.append(get_ll(llm, text[0])) # this has to be a string
    return lls


def process_and_group_tokens(output_tokens):
    tokens = output_tokens.split("<extra_")

    # Initialize an empty list to hold the processed output
    grouped_words = []
    
    for token in tokens:

        
        if token.startswith("<pad>"):
            continue
        else:
            # split token on >, and keep the second half
            parts = token.split(">")
            if len(parts) > 1:  # Ensure there is a second part
                second_token = parts[1].strip()  # Get the second part and strip whitespace
                grouped_words.append(second_token)
                
            
    cleaned_words = [word for word in grouped_words if word != "SPLIT"]

    return cleaned_words


def fill_masks_with_t5(texts, model_name="t5-small", mask_top_p=1.0, first_half = ""):

    filled_texts = []
    
    # Load the tokenizer and model
    if model_name == "t5-large":
        mask_tokenizer = T5tokenizer_large
        mask_model = T5model_large

    elif model_name == "t5-small":
        mask_tokenizer = T5tokenizer_small
        mask_model = T5model_small

    expected = [[x for x in text.split() if x.startswith("<extra_id_")] for text in texts]
    
    inner_list = expected[0]
    ids = [int(token.split('_')[2][:-1]) for token in inner_list]

    if not ids:  # Check if the list is empty
        print('empty list??')
        combined = first_half + texts[0]
        return combined

    min_id = min(ids)
    max_id = max(ids)

    stop_id = mask_tokenizer.encode(f"<extra_id_{max_id}>")
    tokens = mask_tokenizer(
        texts, return_tensors="pt", padding=True
    ) 
    
    outputs = mask_model.generate(
        **tokens,
        max_length=150,
        do_sample=True,
        top_p=mask_top_p,
        num_return_sequences=1,
        eos_token_id=stop_id,
    )
    
    test = mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)
    output_tokens = process_and_group_tokens(test[0])
    
    maxim = min_id + len(output_tokens) 
    text = texts[0]

    # Iterate through the masks in the original text and replace them with the output tokens
    for i in range(min_id, maxim):
        # Replace the corresponding <extra_id_x> with the output token
        text = text.replace(f"<extra_id_{i}>", output_tokens[i])
    

    # fix the weird spacings
    # 1. Replace multiple spaces with a single space
    cleaned_text = re.sub(r"\s+", " ", text)
    # 2. Ensure no spaces before punctuation like commas or periods
    filled_texts = re.sub(r"\s([.,!?])", r"\1", cleaned_text)

    filled_texts = first_half + filled_texts
    start_index = filled_texts.find('<extra_id_')

    
    if start_index != -1:
        # but start at the beginning of the sentence that contains this , not at the word
        start_index = filled_texts.rfind('.', 0, start_index) #+ 1 
        first_half = text[:start_index] 
        second_half = text[start_index:]
        
        # find the extra_id in the text and start counting at 0 again
        second_half = reset_extra_ids(second_half)
        filled_texts = fill_masks_with_t5([second_half], model_name, mask_top_p, first_half = first_half)
        
    return filled_texts

def reset_extra_ids(text):
    # Find all the <extra_id_x> occurrences in the text
    extra_ids = re.findall(r'<extra_id_\d+>', text)

    # Replace them with sequentially numbered <extra_id_> tokens starting from 0
    for i, extra_id in enumerate(extra_ids):
        text = text.replace(extra_id, f"<extra_id_{i}>", 1)  # Replace one at a time

    return text

def replace_masks_llama_cpp(texts, base_model) -> List[str]:
    pattern = re.compile(r"<extra_id_\d+>")
  
    for i in range(len(texts)):
        text = texts[i]  # Sentence from the training set
        output_text = text

        while pattern.search(output_text):
            match = pattern.search(output_text)

            before_id = output_text[: match.start()]
            after_id = output_text[match.end() :]

            # Generate text for the mask
            # how to prevent the model from outputting a number??
            generated_text = base_model(
                prompt=before_id, suffix=after_id, stop=".", max_tokens=2
            )["choices"][0]["text"]
            generated_text = generated_text.replace("\xa0", " ")
            generated_text = generated_text.replace("\u2019", "'")
            
            # 1. Replace multiple spaces with a single space
            cleaned_text = re.sub(r"\s+", " ", generated_text)

            # 2. Ensure no spaces before punctuation like commas or periods
            generated_text = re.sub(r"\s([.,!?])", r"\1", cleaned_text)

            # Update the text by replacing the first [MASK] with the generated text
            output_text = before_id + generated_text

        texts[i] = output_text  # Store the fully replaced text

    return texts


def tokenize_and_mask(text, span_length, pct, ceil_pct=False, max_attempts=5) -> str:
    """PCT is the percentage of tokens in the string that is masked"""

    if isinstance(text, list):
        text = text[0]

    tokens = [token for token in text.split(" ") if token.strip()]

    mask_string = "<<<mask>>>"

    n_spans = pct * len(tokens) / (span_length)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    attempts = 0

    while n_masks < n_spans and attempts < max_attempts:
        start = np.random.randint(1, len(tokens) - span_length)
        end = start + span_length
        search_start = max(1, start)
        search_end = min(len(tokens), end)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
            attempts = 0  # rest affter succesful masking

        else:
            attempts += 1

    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f"<extra_id_{num_filled}>"
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"

    text = " ".join(tokens)

    return text

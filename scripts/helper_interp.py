from transformers import AutoTokenizer, LlamaForCausalLM, Gemma2ForCausalLM
import numpy as np
from typing import List
from transformers import AutoTokenizer
import torch 
import math
import logging
import os
import json
import argparse
import csv
import random
from helper import get_ll, get_lls, tokenize_and_mask, replace_masks_llama_cpp, fill_masks_with_t5

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

disable_tqdm = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_perturbations(model_name, split, n_pertubations=5, alphas = [0.2], span_length = 1, ceil_pct=False) -> List[float]:
    if split == "mini":
        file_path = f'data/{split}.json'
    else:
        file_path = f'data/{model_name}/{split}.json'

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    with open(file_path, 'r') as file:
        data = json.load(file)
        data = data[:5] 
       
    for item in data: 
        for key in ['answer', 'follow-up', 'answer-llm', 'follow-up-llm']:
            if key in item:
                text = item[key]

                if isinstance(text, str):
                    text = [text]

                for alpha in alphas:
                    texts = [text]
                    masked_texts = []
                    for _ in range(n_pertubations):
                        masked_texts.append([tokenize_and_mask(x, span_length, alpha, ceil_pct) for x in texts])

                    item[f"{key}_alpha_{alpha}_{n_pertubations}_noised"] = masked_texts

    if split == "mini":
        output_file_path = f'data/{split}.json'
    else:
        output_file_path = f'data/{model_name}/{split}.json'

    with open(output_file_path, 'w') as file:
        json.dump(data, file, indent=4)

    return


def denoise_with_llm(model_name, mask_model, split, n_pertubations, alphas = [0.2]) -> List[float]:
    """
    Denoises text data using a specified language model.

    Args:
        model_name (str): 
            The name of the model that generated the dataset (e.g., 'human', 'llama', 'gemma').
        
        mask_model (str): 
            The model used for replacing the masked tokens. Options include 'llama', 'T5-small', or 'T5-large'.
        
        split (str): 
            The dataset split to process (e.g., 'train', 'test').
        
        n_pertubations (int, optional): 
            The number of perturbations to create. 
        
        alphas (List[float], optional): 
            A list of alpha values for perturbation severity. Defaults to [0.2].

    Returns:
        List[str]: 
            A list of denoised texts resulting from the application of the mask model to the noised input texts.
    """
    
    if model_name == 'human' or split == 'mini': #mini
        file_path = f'data/{split}.json'

    else: 
        file_path = f'data/{model_name}/{split}.json'

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    with open(file_path, 'r') as file:
        data = json.load(file)
        data = data[:5] 
        
    for item in data: 
        for key in ['answer', 'follow-up', 'answer-llm', 'follow-up-llm']:
            if key in item:
                for alpha in alphas:
                    # Access the noised texts
                    noised_texts = item.get(f"{key}_alpha_{alpha}_{n_pertubations}_noised", []) 

                    logger.info(f"Starting to process noised texts")
                    if mask_model == 'llama':

                        perturbed_texts = replace_masks_llama_cpp(noised_texts, base_model) 
                        
                    elif mask_model == 'T5-small':
                        perturbed_texts = []

                        for text in noised_texts:
                            pertrub = fill_masks_with_t5(text, model_name='t5-small')
                            perturbed_texts.append([pertrub]) 
                       
                    elif mask_model == 'T5-large':
                        perturbed_texts = []
                        for text in noised_texts:
                            pertrub =  fill_masks_with_t5(text, model_name='t5-large')
                            perturbed_texts.append([pertrub]) 

                        
                    item[f"{key}_alpha_{alpha}_{n_pertubations}_{mask_model}_denoised"] = perturbed_texts
                    logger.info("Added filled perturbations")

                    
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Processed filled perturbations, saved to: {file_path}")
    return

def return_scores(detector_model,base_model, split, tokenizer, model_name, mask_model, n_pertubations, alphas, text,device):

    csv_file_path = f'results_detector_{detector_model}.csv'
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        if not file_exists:
            csvwriter.writerow(['Key', 'Alpha', 'N_Perturbations', 'Mask_Model', 'Original_Curvature', 'Normalized_Curvature'])

    if detector_model == 'base_model':
        detector_model = base_model

    if model_name == 'human' or split == 'mini':
        file_path = f'data/{split}.json'

    elif split != 'mini':
        file_path = f'data/{model_name}/{split}.json'

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    with open(file_path, 'r') as file:
        data = json.load(file)
      
    for item in data: 
        for key in ['answer', 'follow-up', 'answer-llm', 'follow-up-llm']:
            if key in item:
                original_text = item[key]

                if isinstance(original_text, str):
                    original_text = [original_text]

                original_ll = get_ll(detector_model, tokenizer, original_text[0])

                for alpha in alphas:
                    # Access the denoised texts - this is a list of perturbations
                    denoised_texts = item.get(f"{key}_alpha_{alpha}_{n_pertubations}_{mask_model}_denoised", [])
                    perturbed_lls = get_lls(detector_model, tokenizer, denoised_texts, disable_tqdm) # if one of theme is None (ignore the following lines)

                    mean_perturbed_lls = np.mean([i for i in perturbed_lls if not math.isnan(i) and i is not None])
                    std_perturbed_lls = np.std([i for i in perturbed_lls if not math.isnan(i)]) if (len([i for i in perturbed_lls if not (math.isnan(i) or 0)]) > 1) else 1

                    curvature_normalized = (original_ll - mean_perturbed_lls) / std_perturbed_lls
                    original_curvature = (original_ll - mean_perturbed_lls)

                    item[f"RESULT_{key}_alpha_{alpha}_{n_pertubations}_{mask_model}_denoised"]  = [str(original_curvature), str(curvature_normalized)]
                    # write to output file with one column original, one normalized curvature add column for key, alpha, n_pertrubations, mask model
                    # output to CSV file
                    # Write to CSV
                    with open(csv_file_path, 'a', newline='') as csvfile:
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow([key,alpha, n_pertubations, mask_model, original_curvature, curvature_normalized])
                               
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
        
        print(f"Processed filled perturbations, saved to: {file_path}")

    return

def mini_detector(detector_model,split,tokenizer,model_name,mask_model,n_pertubations,alphas,base_model,text,device):

    if split == 'mini': 
        print("Making json file for single input text")

        data = {
            "id": 0,
            "answer": text,
        }
        file_path = f'data/{split}.json'
        with open(file_path, 'w') as file:
            json.dump([data], file, indent=4)

    make_perturbations(
        model_name=model_name,
        split=split,
        n_pertubations=n_pertubations,
        alphas=alphas)

    denoise_with_llm(
        model_name=model_name, 
        mask_model=mask_model,
        split=split,
        n_pertubations=n_pertubations,
        alphas=alphas
        )

    return_scores(
        detector_model = 'base_model',
        base_model=base_model,
        split=split,
        tokenizer = tokenizer,
        model_name=model_name,
        mask_model=mask_model,
        n_pertubations=n_pertubations,
        alphas=alphas,
        text=text,
        device=device)

    file_path = f'data/{split}.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

    threshold = -136.23535 # change this accordingly

    for key in data[0]:
        if key.startswith("RESULT_"):
            score = data[0][key][0]

    # return float(score)
    if float(score) < threshold:
        return 0, float(score) # ai-generated
    else:
        return 1, float(score) # human-written


###### RUN THE MAIN #######

if __name__ == "__main__":
    
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process and perturb text data.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model that generated the data.")
    parser.add_argument("--hf_token", type=str, required=True, help="Private huggingface access token.")
    parser.add_argument("--split", type=str, required=True, help="Data split to process (e.g., train, test). Or mini for single sentence.")
    parser.add_argument("--n_perturbations", type=int, default=5, help="Number of perturbations to create.")
    parser.add_argument("--alphas", type=float, nargs='+', default=[0.2], help="List of alpha values for perturbations.")
    parser.add_argument("--span_length", type=int, default=1, help="Length of the span to mask.") # USE 1, not more
    parser.add_argument("--mask_model", type=str, required=True, choices=['llama', 'T5-small', 'T5-large'], help="Model to use for denoising.")
    parser.add_argument("--detector_model", type=str, default="base_model", help="Model to use for detection (defaults to base_model).")
    parser.add_argument("--text", type=str, default=" ", help="A single sentence to detect")


    args = parser.parse_args()

    seed_value = 42
    set_all_seeds(seed_value)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    access_token=args.hf_token

    if args.detector_model == "llama":
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
        base_model = LlamaForCausalLM.from_pretrained(model_id, token=access_token)


    elif args.detector_model == "gemma":
        model_id ="google/gemma-2-9b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
        base_model = Gemma2ForCausalLM.from_pretrained(model_id, token=access_token)

    print("==================================")
    print(f"Model and tokenizer for {args.detector_model} initialized")
    print("==================================")

    base_model = base_model.to(device)
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    if args.text != " ":

        c, score = mini_detector(
            detector_model = base_model,
            split=args.split,
            tokenizer = tokenizer,
            model_name=args.model_name,
            mask_model=args.mask_model,
            n_pertubations=args.n_perturbations,
            alphas=args.alphas,
            base_model=base_model,
            text=args.text,
            device=device
        )
        if c == 0:
            print('The text is AI-generated')
        else:
            print('The text is human-written')

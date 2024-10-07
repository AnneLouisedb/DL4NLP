from transformers import AutoTokenizer, LlamaForCausalLM, Gemma2ForCausalLM
import numpy as np
from tqdm import tqdm
from typing import List
from transformers import AutoTokenizer
import torch 
import math
import logging
import os
import json
from typing import List
import argparse
import csv
import random

# local functions
from helper import get_ll, get_lls, tokenize_and_mask, replace_masks_llama_cpp, fill_masks_with_t5

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

disable_tqdm = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_perturbations(file_path, n_pertubations=5, alphas=[0.2], span_length=1, ceil_pct=False) -> List[float]:
    # Use file_path instead of hardcoded file path

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    with open(file_path, 'r') as file:
        data = json.load(file)
       
    for item in tqdm(data):
        for key in ['answer', 'answer-llm']:
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

    output_file_path = file_path  # Save back to the same file or modify as needed
   
    with open(output_file_path, 'w') as file:
        json.dump(data, file, indent=4)

    return


def denoise_with_llm(mask_model, file_path, n_pertubations, alphas=[0.2]) -> List[float]:
    # Use file_path instead of hardcoded file path

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    with open(file_path, 'r') as file:
        data = json.load(file)
        
    for item in tqdm(data):
        for key in ['answer', 'answer-llm']:
            if key in item:
                for alpha in alphas:
                    # check if this key exisis, else, pass "{key}_alpha_{alpha}_{n_pertubations}_{mask_model}_denoised"
                    denoised_key = f"{key}_alpha_{alpha}_{n_pertubations}_{mask_model}_denoised"
                    # Check if this key already exists, skip if it does
                    if denoised_key in item:
                        logger.info(f"Key {denoised_key} already exists. Skipping processing.")
                        break

                    noised_texts = item.get(f"{key}_alpha_{alpha}_{n_pertubations}_noised", []) 
                    logger.info(f"Starting to process noised texts")
                    if mask_model == 'llama':
                        perturbed_texts = replace_masks_llama_cpp(noised_texts, base_model) 
                    elif mask_model == 'T5-small':
                        perturbed_texts = [fill_masks_with_t5(text, model_name='t5-small') for text in noised_texts]
                    elif mask_model == 'T5-large':
                        perturbed_texts = [fill_masks_with_t5(text, model_name='t5-large') for text in noised_texts]

                    item[f"{key}_alpha_{alpha}_{n_pertubations}_{mask_model}_denoised"] = perturbed_texts
                    logger.info("Added filled perturbations")
        
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
            logger.info(f"Saved processed item to {file_path}")
            

    return

def return_scores(detector_model, tokenizer, model_name, mask_model, file_path, n_pertubations, alphas):
    csv_file_path = f"results_detector_{args.detector_model.split('/')[-1]}.csv"

    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not file_exists:
            csvwriter.writerow(['Key', 'Alpha', 'N_Perturbations', 'Mask_Model', 'Original_Curvature', 'Normalized_Curvature'])

    file_path = file_path

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    with open(file_path, 'r') as file:
        data = json.load(file)
      
    for item in tqdm(data):
        for key in ['answer', 'answer-llm']:
            if key in item:
                original_text = item[key]
                if isinstance(original_text, str):
                    original_text = [original_text]

                original_ll = get_ll(detector_model, tokenizer, original_text[0])

                for alpha in alphas:
                    denoised_texts = item.get(f"{key}_alpha_{alpha}_{n_pertubations}_{mask_model}_denoised", [])

                    denoised_key = f"RESULT_{key}_alpha_{alpha}_{n_pertubations}_model_{model_name}_maskmodel_{args.mask_model}_detectormodel_{args.detector_model}_denoised"
                    # Check if this key already exists, skip if it does
                    if denoised_key in item:
                        logger.info(f"Key {denoised_key} already exists. Skipping processing.")
                        break
                    
                    perturbed_lls = get_lls(detector_model, tokenizer, denoised_texts, disable_tqdm)
                    
                    mean_perturbed_lls = np.mean([i for i in perturbed_lls if not math.isnan(i) and i is not None])
                    std_perturbed_lls = np.std([i for i in perturbed_lls if not math.isnan(i)]) if (len([i for i in perturbed_lls if not (math.isnan(i) or 0)]) > 1) else 1
                    
                    curvature_normalized = (original_ll - mean_perturbed_lls) / std_perturbed_lls
                    original_curvature = (original_ll - mean_perturbed_lls)

                    item[f"RESULT_{key}_alpha_{alpha}_{n_pertubations}_model_{model_name}_maskmodel_{args.mask_model}_detectormodel_{args.detector_model}_denoised"]  = [str(original_curvature), str(curvature_normalized), original_ll]
                    item[f"PERTURBED_LLS_{key}_alpha_{alpha}_{n_pertubations}_model_{model_name}_maskmodel_{args.mask_model}_detectormodel_{args.detector_model}_denoised"]  = perturbed_lls
                    
                    with open(csv_file_path, 'a', newline='') as csvfile:
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow([key, alpha, n_pertubations, mask_model, original_curvature, curvature_normalized])

        if file_path == '/home/scur1744/data/gemma-2-9b-it/test.json':
            out_path = '/home/scur1744/data/gemma-2-9b-it/test.json' 
        elif file_exists == '/home/scur1744/data/Meta-Llama-3.1-8B-Instruct/test.json':
            out_path ='/home/scur1744/data/Meta-Llama-3.1-8B-Instruct/test.json'

        with open(out_path, 'w') as file:
            json.dump(data, file, indent=4)   
        print(f"Processed filled perturbations scores, saved to: {out_path}")

    return

###### RUN THE MAIN #######

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and perturb text data.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model that generated the data.")
    parser.add_argument("--hf_token", type=str, required=True, help="Private huggingface access token.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--n_perturbations", type=int, default=5, help="Number of perturbations to create.")
    parser.add_argument("--alphas", type=float, nargs='+', default=[0.2, 0.3], help="List of alpha values for perturbations.")
    parser.add_argument("--span_length", type=int, default=1, help="Length of the span to mask.")
    parser.add_argument("--make_perturbations", type=bool, default=False, help="Perturbations need to be created")
    parser.add_argument("--denoise_with_llm", type=bool, default=False, help="Text needs to be denoised")
    parser.add_argument("--return_scores", type=bool, default=True, help="Calculating the scores")
    parser.add_argument("--mask_model", type=str, required=True, choices=['llama', 'T5-small', 'T5-large'], help="Model to use for denoising.")
    parser.add_argument("--detector_model", type=str, default="base_model", help="Model to use for detection (defaults to base_model).")

    args = parser.parse_args()

    seed_value = 42
    set_all_seeds(seed_value)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    access_token = args.hf_token

    model_id = args.detector_model
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)

    if "llama" in args.detector_model:
        base_model = LlamaForCausalLM.from_pretrained(model_id, token=access_token)

    if "gemma" in args.detector_model:
        base_model = Gemma2ForCausalLM.from_pretrained(model_id, token=access_token)

    print("==================================")
    print(f"Model and tokenizer for {args.detector_model} initialized")
    print("==================================")

    base_model = base_model.to(device)
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    
    if args.make_perturbations:
        logger.info("Starting perturbation generation...")
        make_perturbations( 
            file_path=args.file_path,
            n_pertubations=args.n_perturbations,
            alphas=args.alphas
        )

    if args.denoise_with_llm:
        logger.info(f"Starting denoising with the LLM.. {args.mask_model}")

        denoise_with_llm(
            mask_model=args.mask_model,
            alphas = args.alphas,
            file_path=args.file_path,
            n_pertubations=args.n_perturbations,
        )

    if args.return_scores:
        logger.info("Processing complete. Now we need to get the log likelihood scores!")

        return_scores(
            detector_model=base_model,
            tokenizer=tokenizer,
            model_name=args.model_name,
            mask_model=args.mask_model,
            file_path=args.file_path,
            n_pertubations=args.n_perturbations,
            alphas=args.alphas
        )

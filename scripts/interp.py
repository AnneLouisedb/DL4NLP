import torch
import numpy as np
from typing import *
from transformers import AutoTokenizer
from captum.attr import visualization
from transformers import AutoTokenizer, LlamaForCausalLM, Gemma2ForCausalLM
import numpy as np
from transformers import AutoTokenizer
import torch 
import re
import logging
import argparse
import random
from itertools import chain, combinations
from math import factorial
from helper_interp import mini_detector


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

disable_tqdm = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_attribution(sen, attributions,split="mini", interp='sentence', method='shap', model="llama"):
    attributions = attributions.squeeze()

    if True:
        attributions /= torch.max(attributions.abs())

    if interp=="word":
        raw_html = visualization.format_word_importances(
            sen.split(), attributions
        )
    else:
        raw_html = visualization.format_word_importances(
            split_paragraph_into_sentences(sen), attributions
        )

    if split=="mini":
        html_file_path= f'data/{split}_{interp}_{method}_{model}.html'
    else:
        html_file_path = f'data/{split}_{interp}_{method}_{model}.html'

    with open(html_file_path, "w", encoding="utf-8") as file:
        file.write(raw_html)
        print(f"Interpretability visualization saved at {html_file_path}")

def split_paragraph_into_sentences(paragraph):
    sentences = re.split(r'(?<=[.!?]) +', paragraph)
    return sentences

def feature_ablation_word(detector_model,
            split,
            tokenizer,
            model_name,
            mask_model,
            n_perturbations,
            alphas,
            base_model,
            text,
            device):

    a = []
    input_ids = tokenizer.encode(args.text, return_tensors='pt').to(device)
    sen_len = len(input_ids[0])
    baseline = '<<<mask>>>'
    value=tokenizer.encode(baseline, return_tensors='pt')[0][2]
    baseline_ids = torch.full((sen_len,), value, dtype=torch.long)

    print('For text:', args.text)

    _,Mx = mini_detector(
            detector_model = base_model,
            split=split,
            tokenizer = tokenizer,
            model_name=model_name,
            mask_model=mask_model,
            n_pertubations=n_perturbations,
            alphas=alphas,
            base_model=base_model,
            text=text,
            device=device
            )

    for i in range(1,sen_len):
        print('Feature ablation at iteration',i)

        copy = torch.clone(input_ids)
        

        copy[0][i] = baseline_ids[i]
        text_modified = tokenizer.batch_decode(copy,skip_special_tokens=True)
        _,MFx = mini_detector(
            detector_model = base_model,
            split=split,
            tokenizer = tokenizer,
            model_name=model_name,
            mask_model=mask_model,
            n_pertubations=n_perturbations,
            alphas=alphas,
            base_model=base_model,
            text=text_modified,
            device=device
            )

        a.append(Mx - MFx)

    return torch.Tensor(a)

def feature_ablation_sentence(detector_model,
            split,
            tokenizer,
            model_name,
            mask_model,
            n_perturbations,
            alphas,
            base_model,
            text,
            device):

    a = []

    print('For text:', text)

    sentences = split_paragraph_into_sentences(text)
    sen_len = len(sentences)

    _,Mx = mini_detector(
            detector_model = base_model,
            split=split,
            tokenizer = tokenizer,
            model_name=model_name,
            mask_model=mask_model,
            n_pertubations=n_perturbations,
            alphas=alphas,
            base_model=base_model,
            text=text,
            device=device
            )


    for i in range(sen_len):
        print('Feature ablation at iteration', i)

        copy = sentences.copy()
        copy.pop(i)
        text_modified = ' '.join(copy)

        _,MFx = mini_detector(
            detector_model = base_model,
            split=split,
            tokenizer = tokenizer,
            model_name=model_name,
            mask_model=mask_model,
            n_pertubations=n_perturbations,
            alphas=alphas,
            base_model=base_model,
            text=text_modified,
            device=device
            )

        a.append(Mx - MFx)

    return torch.Tensor(a)



def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def coalitions_without_i(sen_len, i):
    features_not_i = [j for j in range(sen_len) if j != i]
    for coalition in powerset(features_not_i):
        yield coalition

def i_to_sentence(sentences, input_ids, value):

    input_ids = input_ids.tolist()
    selected_sentences = []
    for i in input_ids:
        if i != value:
            selected_sentences.append(sentences[i])
    
    return selected_sentences

def compute_coalition_logits(detector_model,
            split,
            tokenizer,
            model_name,
            mask_model,
            n_perturbations,
            alphas,
            base_model,
            text,
            device):
    """
    Computes the model output for all possible coalitions, in batches.

    Returns
    -------
    coalition_logits : torch.Tensor
        shape: (2**sen_len,)
        The output logit for each coalition
    coalition_to_index : Dict[Tuple[int], int]
        Dictionary mapping a coalition tuple to its index in the
        coalition_logits tensor.
    """
    sentences = split_paragraph_into_sentences(text)  
    sen_len = len(sentences)
    input_ids = torch.arange(0, sen_len)
    num_coalitions = 2**sen_len

    baseline = '<<<mask>>>'
    value = tokenizer.encode(baseline, return_tensors='pt')[0][2]
    baseline_ids = torch.full((sen_len,), value, dtype=torch.long)

    coalitions = torch.zeros(num_coalitions, sen_len, dtype=torch.long) #.to(device)
    coalitions += baseline_ids  

    coalition_to_index = {}

    for idx, coalition in enumerate(powerset(range(sen_len))):
        coalition_to_index[coalition] = idx
        if len(coalition) > 0:
            coalition = list(coalition)
            coalitions[idx, coalition] = input_ids[coalition]

    coalition_logits = torch.zeros(num_coalitions)
    batch_size = 1024
    batch_iterator = torch.split(coalitions, batch_size)

    for batch_idx, batch in enumerate(batch_iterator):
        batch_logits = []
        for t in range(batch.shape[0]):
            text_modified = i_to_sentence(sentences, batch[t], value)
            if text_modified:
                _, batch_logit = mini_detector(
                    detector_model = base_model,
                    split=split,
                    tokenizer = tokenizer,
                    model_name=model_name,
                    mask_model=mask_model,
                    n_pertubations=n_perturbations,
                    alphas=alphas,
                    base_model=base_model,
                    text=text_modified,
                    device=device
                    )
                batch_logits.append(batch_logit)
            else:
                batch_logits.append(0)
        
        batch_logits = torch.Tensor(batch_logits)

        coalition_logits[
            batch_idx * batch_size : (batch_idx + 1) * batch_size
        ] = batch_logits

    return coalition_logits, coalition_to_index


def shap_sentence(detector_model,
            split,
            tokenizer,
            model_name,
            mask_model,
            n_perturbations,
            alphas,
            base_model,
            text,
            device):

    print('For text:', text)

    sentences = split_paragraph_into_sentences(text) 
    sen_len = len(sentences)

    coalition_logits, coalition_to_index = compute_coalition_logits(
            detector_model = base_model,
            split=split,
            tokenizer = tokenizer,
            model_name=model_name,
            mask_model=mask_model,
            n_perturbations=n_perturbations,
            alphas=alphas,
            base_model=base_model,
            text=text,
            device=device
            )

    shapley_values = torch.zeros(sen_len)

    for i in range(sen_len):
        for S in coalitions_without_i(sen_len, i):
            S_i = tuple(sorted(S + (i,)))
            index_S = coalition_to_index[S]
            index_S_i = coalition_to_index[S_i]

            p = (factorial(len(S)) * factorial(sen_len - 1 - len(S))) / factorial(sen_len)
            sum = p * (coalition_logits[index_S_i] - coalition_logits[index_S])
            shapley_values[i] += sum
    
    return shapley_values


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process and perturb text data.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model that generated the data.")
    parser.add_argument("--hf_token", type=str, required=True, help="Private huggingface access token.")
    parser.add_argument("--split", type=str, required=True, help="Data split to process (e.g., train, test). Or mini for single sentence.")
    parser.add_argument("--n_perturbations", type=int, default=5, help="Number of perturbations to create.")
    parser.add_argument("--alphas", type=float, nargs='+', default=[0.2], help="List of alpha values for perturbations.")
    parser.add_argument("--span_length", type=int, default=1, help="Length of the span to mask.") 
    parser.add_argument("--mask_model", type=str, required=True, choices=['llama', 'T5-small', 'T5-large'], help="Model to use for denoising.")
    parser.add_argument("--detector_model", type=str, default="base_model", help="Model to use for detection (defaults to base_model).")
    parser.add_argument("--interp", type=str, default="sentence", help="Kind of interpretability to perform (word or sentence)")
    parser.add_argument("--attribution", type=str, default="ablation", help="Kind of attribution method to use (e.g. ablation, shap).")
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

    if args.interp == "word":
        a = feature_ablation_word(
                detector_model = base_model,
                split=args.split,
                tokenizer = tokenizer,
                model_name=args.model_name,
                mask_model=args.mask_model,
                n_perturbations=args.n_perturbations,
                alphas=args.alphas,
                base_model=base_model,
                text=args.text,
                device=device
                )
        visualize_attribution(args.text, a, split=args.split, interp=args.interp, method=args.attribution, model=args.model_name)
    elif args.interp == "sentence":
        if args.attribution == "ablation":
            a = feature_ablation_sentence(
                    detector_model = base_model,
                    split=args.split,
                    tokenizer = tokenizer,
                    model_name=args.model_name,
                    mask_model=args.mask_model,
                    n_perturbations=args.n_perturbations,
                    alphas=args.alphas,
                    base_model=base_model,
                    text=args.text,
                    device=device
                    )
        elif args.attribution == "shap":
            a = shap_sentence(
                    detector_model = base_model,
                    split=args.split,
                    tokenizer = tokenizer,
                    model_name=args.model_name,
                    mask_model=args.mask_model,
                    n_perturbations=args.n_perturbations,
                    alphas=args.alphas,
                    base_model=base_model,
                    text=args.text,
                    device=device
                    )

        visualize_attribution(args.text, a, args.split, args.interp, method=args.attribution, model=args.model_name)

    


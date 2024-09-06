import json
import os
from tqdm import tqdm
from collections import OrderedDict
from typing import Dict, List, Tuple
from utils import OpenAIModel, HuggingFaceModel
from huggingface_hub import login
import argparse

class Model_Baseline:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = os.path.join(args.save_path, args.dataset_name)
        self.demonstration_path = args.demonstration_path
        self.mode = args.mode
        self.api_key = args.api_key
      
        login(token=args.api_key)
        if args.framework == "openai":
            self.model = OpenAIModel(
                args.model_name, args.stop_words, args.max_new_tokens
            )
        elif args.framework == "huggingface":
            self.model = HuggingFaceModel(
                args.api_key,
                args.model_name,
                args.stop_words,
                args.max_new_tokens,
                args.is_GGUF,
                args.Q_type,
            )
        else:
            raise ValueError(
                "Invalid framework. Please choose from [openai, huggingface]"
            )

        self.prompt_creator = self.prompt
        self.model_name = self.model_name.replace("/", "-")
        # save outputs
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def prompt(self, test_example):
        question = test_example["question"].strip()
        return question

    def load_raw_dataset(self, split):
        with open(
            os.path.join(self.data_path, self.dataset_name, f"{split}.json")
        ) as f:
            raw_dataset = json.load(f)
        return raw_dataset

    def reasoning_graph_generation(self):
      """" This function needs to change"""
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")
        
        outputs = []
        for example in tqdm(raw_dataset):
            question = example["question"]

            # create prompt
            full_prompt = self.prompt_creator(example)
            output = self.model.generate(full_prompt)
            if self.args.stop_words:
                output = output[: -len(self.args.stop_words)]
            # create output
            dict_output = self.update_answer(example, output)
            outputs.append(dict_output)

        # save outputs
        with open(
            os.path.join(
                self.save_path,
                f"{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}.json",
            ),
            "w",
        ) as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    def batch_reasoning_graph_generation(self, batch_size=10):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        
        outputs = []
        # split dataset into chunks
        dataset_chunks = [
            raw_dataset[i : i + batch_size]
            for i in range(0, len(raw_dataset), batch_size)
        ]
        for chunk in tqdm(dataset_chunks):
            # create prompt
            full_prompts = [
                self.prompt_creator(example) for example in chunk
            ]
            try:
                batch_outputs = self.model.batch_generate(full_prompts)
                # create output
                for sample, output in zip(chunk, batch_outputs):
                    if self.args.stop_words in output:
                        output = output[: -len(self.args.stop_words)]
                    # get the answer
                    dict_output = self.update_answer(sample, output)
                    outputs.append(dict_output)
            except:
                # generate one by one if batch generation fails
                for sample, full_prompt in zip(chunk, full_prompts):
                    try:
                        output = self.model.generate(full_prompt)
                        if self.args.stop_words in output:
                            output = output[: -len(self.args.stop_words)]
                        # get the answer
                        dict_output = self.update_answer(sample, output)
                        outputs.append(dict_output)
                    except Exception as e:
                        print("Error in generating example:", sample["id"])
                        print("Error message:", str(e))

        # save outputs
        with open(
            os.path.join(
                self.save_path,
                f"{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}.json",
            ),
            "w",
        ) as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    def update_answer(self, sample, output):

        dict_output = {
            "id": sample["id"],
            "question": sample["question"],
            "answer": sample["answer"],
          "model_answer": output   
        }
        return dict_output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./src/data")
    parser.add_argument("--dataset_name", type=str) # default is FollowupQG
    parser.add_argument("--split", type=str)
    parser.add_argument("--save_path", type=str, default="./src/outputs/baselines")
    parser.add_argument(
        "--demonstration_path", type=str, default="./src/models/prompts"
    )
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--framework", type=str, default="openai")
    parser.add_argument("--stop_words", type=str, default="------\n")
    parser.add_argument("--mode", type=str)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--is_GGUF", action="store_true", default=False)
    parser.add_argument("--Q_type", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    model_baseline = Model_Baseline(args)
    model_baseline.reasoning_graph_generation()

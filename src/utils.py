import backoff  # for exponential backoff
import openai
import os
import asyncio
from typing import Any

from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

from llama_cpp import Llama


# Backoff decorator -> Retry function when RateLimitError is raised
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


async def dispatch_openai_chat_requests(
    messages_list: list[list[dict[str, Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str],
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.

    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
        stop_words: List of words to stop the model from generating.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop_words,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


async def dispatch_openai_prompt_requests(
    messages_list: list[list[dict[str, Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str],
) -> list[str]:
    async_responses = [
        openai.Completion.acreate(
            model=model,
            prompt=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop_words,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


class OpenAIModel:
    def __init__(self, API_KEY, model_name, stop_words, max_new_tokens) -> None:
        openai.api_key = API_KEY
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words

    # used for chat-gpt and gpt-4
    def chat_generate(self, input_string, temperature=0.0):
        response = chat_completions_with_backoff(
            model=self.model_name,
            messages=[{"role": "user", "content": input_string}],
            max_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            stop=self.stop_words,
        )
        generated_text = response["choices"][0]["message"]["content"].strip()
        return generated_text

    # used for text/code-davinci
    def prompt_generate(self, input_string, temperature=0.0):
        response = completions_with_backoff(
            model=self.model_name,
            prompt=input_string,
            max_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=self.stop_words,
        )
        generated_text = response["choices"][0]["text"].strip()
        return generated_text

    def generate(self, input_string, temperature=0.0):
        if self.model_name in [
            "text-davinci-002",
            "code-davinci-002",
            "text-davinci-003",
        ]:
            return self.prompt_generate(input_string, temperature)
        elif self.model_name in ["gpt-4", "gpt-3.5-turbo"]:
            return self.chat_generate(input_string, temperature)
        else:
            raise Exception("Model name not recognized")

    def batch_chat_generate(self, messages_list, temperature=0.0):
        open_ai_messages_list = []
        for message in messages_list:
            open_ai_messages_list.append([{"role": "user", "content": message}])
        predictions = asyncio.run(
            dispatch_openai_chat_requests(
                open_ai_messages_list,
                self.model_name,
                temperature,
                self.max_new_tokens,
                1.0,
                self.stop_words,
            )
        )
        return [x["choices"][0]["message"]["content"].strip() for x in predictions]

    def batch_prompt_generate(self, prompt_list, temperature=0.0):
        predictions = asyncio.run(
            dispatch_openai_prompt_requests(
                prompt_list,
                self.model_name,
                temperature,
                self.max_new_tokens,
                1.0,
                self.stop_words,
            )
        )
        return [x["choices"][0]["text"].strip() for x in predictions]

    def batch_generate(self, messages_list, temperature=0.0):
        if self.model_name in [
            "text-davinci-002",
            "code-davinci-002",
            "text-davinci-003",
        ]:
            return self.batch_prompt_generate(messages_list, temperature)
        elif self.model_name in ["gpt-4", "gpt-3.5-turbo"]:
            return self.batch_chat_generate(messages_list, temperature)
        else:
            raise Exception("Model name not recognized")

    def generate_insertion(self, input_string, suffix, temperature=0.0):
        response = completions_with_backoff(
            model=self.model_name,
            prompt=input_string,
            suffix=suffix,
            temperature=temperature,
            max_tokens=self.max_new_tokens,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        generated_text = response["choices"][0]["text"].strip()
        return generated_text


class StoppingCriteriaToken(StoppingCriteria):
    def __init__(self, stop_token_id):
        super().__init__()
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores):
        return input_ids[0][-1] == self.stop_token_id


class HuggingFaceModel:
    def __init__(
        self,
        API_KEY,
        model_name,
        stop_words,
        max_new_tokens,
        is_GGUF=False,
        Q_type=None,
        temperature=0.0,
    ) -> None:
        self.api_key = API_KEY
        self.model_name = model_name
        self.stop_words = stop_words
        self.max_new_tokens = max_new_tokens
        self.is_GGUF = is_GGUF
        self.Q_type = Q_type
        self.temperature = temperature

        if self.is_GGUF:
            self.pipe = Llama.from_pretrained(
                repo_id=self.model_name,
                filename=f"*{self.Q_type}.gguf",
                verbose=False,
                n_gpu_layers=-1,
                n_ctx=2048,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map="auto", torch_dtype="auto"
            )

            stop_token_ids = [
                self.tokenizer(
                    stop_token, return_tensors="pt", add_special_tokens=False
                )["input_ids"].squeeze()
                for stop_token in stop_words.split(" ")
            ]
            stopping_criteria = StoppingCriteriaList(
                [
                    StoppingCriteriaToken(stop_token_id)
                    for stop_token_id in stop_token_ids
                ]
            )

            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                token=self.api_key,
                max_new_tokens=max_new_tokens,
                top_p=1.0,
                device_map="auto",
                do_sample=False,
                return_full_text=False,
                stopping_criteria=stopping_criteria,
            )

            if self.pipe.tokenizer.pad_token_id is None:
                self.pipe.tokenizer.pad_token_id = self.pipe.model.config.eos_token_id

    def generate(self, input_string):
        if self.is_GGUF:
            response = self.pipe(
                input_string,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=1.0,
                echo=False,
                stop=self.stop_words,
            )
            generated_text = response[0]["generated_text"].strip()
        else:
            response = self.pipe(
                input_string,
            )
            generated_text = response[0]["generated_text"].strip()
        return generated_text

    def batch_generate(self, input_strings):
        if self.is_GGUF:
            raise Exception(
                "Batch generation not (yet) supported for GGUF models (using llama-cpp-python)"
            )
        else:
            responses = self.pipe(
                input_strings,
            )
            generated_text = [
                response[0]["generated_text"].strip() for response in responses
            ]
        return generated_text

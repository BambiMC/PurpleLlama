# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import json
import logging

import os
import textwrap
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import requests
from openai._types import NOT_GIVEN

from typing_extensions import override

from .benchmark_utils import audio_to_b64, get_mime_type, image_to_b64
from .cache_handler import CacheHandler

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


import os
import time
from typing import Optional, Sequence, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import openai
from google import genai
from google.genai import types



NUM_LLM_RETRIES = 100
MAX_RETRY_TIMEOUT = 600

LOG: logging.Logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9


class RetriesExceededException(Exception):
    def __init__(
        self, message: str = "Maximum number of retries has been exceeded"
    ) -> None:
        self.message = message
        super().__init__(self.message)


class UnretriableQueryException(Exception):
    pass


@dataclass
class LLMConfig:
    model: str
    api_key: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    cot_end_tokens: list[str] | None = None


class LLM(ABC):
    def __init__(
        self,
        config: LLMConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model: str = config.model
        self.api_key: str | None = config.api_key
        self.cache_handler: CacheHandler | None = None
        self.max_tokens: int = config.max_tokens
        self.cot_end_tokens: list[str] | None = config.cot_end_tokens
        if self.model not in self.valid_models():
            LOG.warning(
                f"{self.model} is not in the valid model list for {type(self).__name__}. Valid models are: {', '.join(self.valid_models())}."
            )

    @abstractmethod
    def query(
        self,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        """
        Abstract method to query an LLM with a given prompt and return the response.

        Args:
            prompt (str): The prompt to send to the LLM.
            guided_decode_json_schema (Optional[str]): JSON schema for guided decoding.
            temperature (float): Controls randomness in the output.
            top_p (float): Controls diversity via nucleus sampling.

        Returns:
            str: The response from the LLM.
        """
        pass

    @abstractmethod
    def chat(
        self,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        """
        Abstract method to chat with an LLM with given prompts and return the response.

        Args:
            prompt_with_history ([str]): The prompt to send to the LLM along with chat history. The last string in the list is the prompt.
            guided_decode_json_schema (Optional[str]): JSON schema for guided decoding.
            temperature (float): Controls randomness in the output.
            top_p (float): Controls diversity via nucleus sampling.

        Returns:
            str: The response from the LLM.
        """
        pass

    @abstractmethod
    def chat_with_system_prompt(
        self,
        system_prompt: str,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        """
        Abstract method to chat with an LLM with given prompts as well as system prompt and return the response.

        Args:
            system_prompt (str): The system prompt to send to the LLM.
            prompt_with_history ([str]): The prompt to send to the LLM along with chat history. The last string in the list is the prompt.
            guided_decode_json_schema (Optional[str]): JSON schema for guided decoding.
            temperature (float): Controls randomness in the output.
            top_p (float): Controls diversity via nucleus sampling.

        Returns:
            str: The response from the LLM.
        """
        pass

    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        """
        Abstract method to query an LLM with a given prompt and return the response.

        Args:
            system prompt (str): The system prompt to send to the LLM.
            prompt (str): The prompt to send to the LLM.
            guided_decode_json_schema (Optional[str]): JSON schema for guided decoding.
            temperature (float): Controls randomness in the output.
            top_p (float): Controls diversity via nucleus sampling.

        Returns:
            str: The response from the LLM.
        """
        return self.query(
            system_prompt + "\n" + prompt, guided_decode_json_schema, temperature, top_p
        )

    def _query_with_retries(
        self,
        func: Callable[..., str],
        *args: Union[str, bool, Optional[str], float, float],
        retries: int = NUM_LLM_RETRIES,
        max_retry_timeout: int = MAX_RETRY_TIMEOUT,
        backoff_factor: float = 0.5,
    ) -> str:
        last_exception = None
        for retry in range(retries):
            try:
                return func(*args)
            except Exception as exception:
                last_exception = exception
                sleep_time = (
                    backoff_factor * (2**retry)
                    if backoff_factor * (2**retry) <= max_retry_timeout
                    else max_retry_timeout
                )
                time.sleep(sleep_time)
                LOG.error(
                    f"LLM Query failed with error {type(exception)}: {textwrap.shorten(str(exception), width=250)}. Sleeping for {sleep_time} seconds..."
                )
        raise RetriesExceededException(
            f"Unable to query LLM after {retries} retries: {last_exception}"
        )

    def _chat_with_retries(
        self,
        func: Callable[..., str],
        *args: Union[List[str], bool, Optional[str], float, float],
        retries: int = NUM_LLM_RETRIES,
        max_retry_timeout: int = MAX_RETRY_TIMEOUT,
        backoff_factor: float = 0.5,
    ) -> str:
        last_exception = None
        for retry in range(retries):
            try:
                return func(*args)
            except Exception as exception:
                last_exception = exception
                sleep_time = (
                    backoff_factor * (2**retry)
                    if backoff_factor * (2**retry) <= max_retry_timeout
                    else max_retry_timeout
                )
                time.sleep(sleep_time)
                LOG.error(
                    f"LLM Chat query failed with error: {exception}. Sleeping for {sleep_time} seconds..."
                )
        raise RetriesExceededException(
            f"Unable to query LLM after {retries} retries: {last_exception}"
        )

    def query_with_retries(
        self,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        if self.cache_handler:
            cached_response = self.cache_handler.get_response(self.model, None, prompt)
            if cached_response:
                return cached_response
        response = self._query_with_retries(
            self.query, prompt, guided_decode_json_schema, temperature, top_p
        )

        if self.cache_handler:
            self.cache_handler.save_response(self.model, None, prompt, response)

        return response

    def chat_with_retries(
        self,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        if self.cache_handler:
            cached_response = self.cache_handler.get_response_chat(
                self.model, None, prompt_with_history
            )
            if cached_response:
                return cached_response
        response = self._chat_with_retries(
            self.chat,
            prompt_with_history,
            guided_decode_json_schema,
            temperature,
            top_p,
        )
        if self.cache_handler:
            self.cache_handler.save_response_chat(
                self.model, None, prompt_with_history, response
            )
        return response

    def query_with_system_prompt_with_retries(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        if self.cache_handler:
            cached_response = self.cache_handler.get_response(
                self.model, system_prompt, prompt
            )
            if cached_response:
                return cached_response
        response = self._query_with_retries(
            self.query_with_system_prompt,
            system_prompt,
            prompt,
            guided_decode_json_schema,
            temperature,
            top_p,
        )

        if self.cache_handler:
            self.cache_handler.save_response(
                self.model, system_prompt, prompt, response
            )

        return response

    def chat_with_system_prompt_with_retries(
        self,
        system_prompt: str,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        if self.cache_handler:
            cached_response = self.cache_handler.get_response_chat(
                self.model, system_prompt, prompt_with_history
            )
            if cached_response:
                return cached_response
        response = self._chat_with_retries(
            self.chat_with_system_prompt,
            system_prompt,
            prompt_with_history,
            guided_decode_json_schema,
            temperature,
            top_p,
        )
        if self.cache_handler:
            self.cache_handler.save_response_chat(
                self.model, system_prompt, prompt_with_history, response
            )
        return response

    def query_with_system_prompt_and_image(
        self,
        system_prompt: str,
        prompt: str,
        image_path: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        """
        Abstract method to query an LLM with a given prompt, system prompt, and image and return the response.
        """
        LOG.error(
            "[ERROR] Please override this method in a subclass to use this method."
        )
        return "Query API not implemented yet."

    def query_with_system_prompt_and_image_with_retries(
        self,
        system_prompt: str,
        prompt: str,
        image_path: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        return self._query_with_retries(
            self.query_with_system_prompt_and_image,
            system_prompt,
            prompt,
            image_path,
            guided_decode_json_schema,
            temperature,
            top_p,
        )

    def query_multimodal(
        self,
        system_prompt: Optional[str] = None,
        text_prompt: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        audio_paths: Optional[List[str]] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        """
        Abstract method to query an LLM and return the response. This method
        supports multimodal queries with text, images and audio, and usage of a system prompt.

        Args:
            system_prompt (Optional[str]): The system prompt to send to the LLM.
            text_prompt (Optional[str]): The prompt to send to the LLM.
            image_paths (Optional[List[str]]): The paths to the images to send to the LLM.
            audio_paths (Optional[List[str]]): The paths to the audio files to send to the LLM.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Controls randomness in the output.
            top_p (float): Controls diversity via nucleus sampling.

        Returns:
            Optional[str]: The response from the LLM, or None if it failed to query.
        """
        raise NotImplementedError(
            "Please override this method in a subclass to use this method."
        )

    def query_multimodal_with_retries(
        self,
        system_prompt: Optional[str] = None,
        text_prompt: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        audio_paths: Optional[List[str]] = None,
        num_retries: int = NUM_LLM_RETRIES,
        max_retry_timeout: int = MAX_RETRY_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        backoff_factor: float = 0.5,
    ) -> Optional[str]:
        """
        Abstract method to query an LLM and return the response, wrapped in retries.
        This method supports multimodal queries and system prompt.

        Args:
            system_prompt (Optional[str]): The system prompt to send to the LLM.
            text_prompt (Optional[str]): The prompt to send to the LLM.
            image_paths (Optional[List[str]]): The paths to the images to send to the LLM.
            audio_paths (Optional[List[str]]): The paths to the audio files to send to the LLM.
            num_retries (int): Number of retries to attempt.
            max_retry_timeout (int): Maximum timeout between retries.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Controls randomness in the output.
            top_p (float): Controls diversity via nucleus sampling.
            backoff_factor (float): Factor to increase backoff time between retries.

        Returns:
            Optional[str]: The response from the LLM, or None if it failed to query.
        """
        cache_key = json.dumps(
            {
                "text_prompt": text_prompt,
                "image_paths": (
                    sorted(image_paths) if image_paths is not None else None
                ),
                "audio_paths": (
                    sorted(audio_paths) if audio_paths is not None else None
                ),
            }
        )
        if self.cache_handler:
            cached_response = self.cache_handler.get_response(
                self.model, system_prompt, cache_key
            )
            if cached_response:
                return cached_response

        last_exception = None
        for retry in range(num_retries):
            try:
                response = self.query_multimodal(
                    system_prompt=system_prompt,
                    text_prompt=text_prompt,
                    image_paths=image_paths,
                    audio_paths=audio_paths,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                if self.cache_handler:
                    self.cache_handler.save_response(
                        self.model, system_prompt, cache_key, response
                    )
                return response
            except NotImplementedError as exception:
                LOG.error(
                    f"LLM Query failed as the method was not implemented: {exception}"
                )
                raise exception
            except UnretriableQueryException as exception:
                LOG.error(f"LLM Query failed due to unretriable error: {exception}")
                return None
            except Exception as exception:
                last_exception = exception
                sleep_time = (
                    backoff_factor * (2**retry)
                    if backoff_factor * (2**retry) <= max_retry_timeout
                    else max_retry_timeout
                )
                time.sleep(sleep_time)
                LOG.error(
                    f"LLM Query failed with error: {exception}. Sleeping for {sleep_time} seconds..."
                )
        LOG.error(f"Unable to query LLM after {num_retries} retries: {last_exception}")
        return None

    def valid_models(self) -> list[str]:
        """List of valid model parameters, e.g. 'gpt-3.5-turbo' for GPT"""
        return []


SPECIFICATION_FORMAT: str = "<PROVIDER>::<MODEL>(::<API KEY>)?"
EXAMPLE_SPECIFICATION: str = "OPENAI::gpt-3.5-turbo::<YOUR API KEY>"


def create(
    identifier: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    cot_end_tokens: List[str] | None = None,
) -> LLM:
    """
    Takes a string of the format <SPECIFICATION_FORMAT> and generates LLM instance.
    See <EXAMPLE_SPECIFICATION> for example.
    """
    api_key: str | None = None
    split = identifier.split("::")
    if len(split) == 2:
        provider, name = split
    elif len(split) == 3:
        provider, name, api_key = split
    else:
        raise ValueError(f"Invalid identifier: {identifier}")

    if api_key is None:
        raise ValueError(f"No API key provided for {provider}: {name}")

    config = LLMConfig(
        model=name,
        api_key=api_key,
        max_tokens=max_tokens,
        cot_end_tokens=cot_end_tokens,
    )

    if provider == "OPENAI":
        return OPENAI(config)
    if provider == "TOGETHER":
        return TOGETHER(config)
    if provider == "ANTHROPIC":
        return ANTHROPIC(config)
    if provider == "GOOGLEGENAI":
        return GOOGLEGENAI(config)
    if provider == "DEEPSEEK":
        return DEEPSEEK(config)
    if provider == "LOCALLLM":
        return LOCALLLM(name, True, True, "cuda")

    raise ValueError(f"Unknown provider: {provider}")


class OPENAI(LLM):
    """Accessing OpenAI, updated with CloudModel-style logic"""

    def __init__(self, config: LLMConfig, timeout: float = 900.0) -> None:
        self._reasoning_models = [
            "o1", "o1-mini", "o3", "o3-mini", "o4-mini",
            "gpt-5-nano", "gpt-4.1-nano"
        ]
        super().__init__(config)
        self.client = openai.OpenAI()
        self.timeout = timeout

    @override
    def valid_models(self) -> list[str]:
        return ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4-turbo"] + self._reasoning_models

    @override
    def chat(self, prompt_with_history: list[str], guided_decode_json_schema: str = None,
             temperature: float = DEFAULT_TEMPERATURE, top_p: float = DEFAULT_TOP_P) -> str:
        raise NotImplementedError("Not implemented yet for OPENAI.")

    @override
    def chat_with_system_prompt(self, system_prompt: str, prompt_with_history: list[str],
                                guided_decode_json_schema: str = None,
                                temperature: float = DEFAULT_TEMPERATURE,
                                top_p: float = DEFAULT_TOP_P) -> str:
        raise NotImplementedError("Not implemented yet for OPENAI.")


    @override
    def query(self, prompt: str, guided_decode_json_schema: str = None,
              temperature: float = DEFAULT_TEMPERATURE, top_p: float = DEFAULT_TOP_P) -> str:
        """
        Sends a single prompt to the model and returns the response text.
        """

        try:
            print(f"Query:\n{prompt}")
            response = self.client.with_options(timeout=self.timeout).responses.create(
                model=self.model,
                # instructions="",  # You can add a default system prompt here if needed
                input=prompt,
                # temperature=temperature,
                # top_p=top_p,
                # max_output_tokens can be set if required
            )
            print(f"Response:\n{response.output_text}")
            return response.output_text
        except Exception as e:
            print(f"OpenAI request failed: {e}")
            return ""

    @override
    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        """
        Sends a single prompt to the model with a system prompt.
        """
        try:
            print(f"Query:\n{prompt}")
            response = self.client.with_options(timeout=self.timeout).responses.create(
                model=self.model,
                instructions=system_prompt,
                input=prompt,
                # temperature=temperature,
                # top_p=top_p,
                # max_output_tokens can be set if needed
            )
            print(f"Response:\n{response.output_text}")
            return response.output_text
        except Exception as e:
            print(f"OpenAI request failed: {e}")
            return ""

    @override
    def query_with_retries(
        self,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 3,
        backoff: float = 1.5,
    ) -> str:
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Query:\n{prompt}")
                response = self.client.with_options(timeout=self.timeout).responses.create(
                    model=self.model,
                    instructions="",
                    input=prompt,
                    # temperature=temperature,
                    # top_p=top_p,
                    # max_output_tokens=max_new_tokens,
                )
                print(f"Response:\n{response.output_text}")
                return response.output_text
            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    time.sleep(backoff ** attempt)
        return ""

    @override
    def query_with_system_prompt_with_retries(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 3,
        backoff: float = 1.5,
    ) -> str:
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Query:\n{prompt}")
                response = self.client.with_options(timeout=self.timeout).responses.create(
                    model=self.model,
                    instructions=system_prompt,
                    input=prompt,
                    # temperature=temperature,
                    # top_p=top_p,
                    # max_output_tokens=max_new_tokens,
                )
                print(f"Response:\n{response.output_text}")
                return response.output_text
            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    time.sleep(backoff ** attempt)
        return ""

    @override
    def query_with_system_prompt_and_image(
        self,
        system_prompt: str,
        prompt: str,
        image_path: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        raise NotImplementedError("Not implemented yet for OPENAI.")

    @override
    def query_multimodal(
        self,
        system_prompt: Optional[str] = None,
        text_prompt: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        audio_paths: Optional[List[str]] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        raise NotImplementedError("Not implemented yet for OPENAI.")



class TOGETHER(LLM):
    """Accessing TOGETHER"""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.together.xyz/v1",
        )

    @override
    def chat(
        self,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = 0.75,
        top_p: float = 1,
    ) -> str:
        messages = []
        for i in range(len(prompt_with_history)):
            if i % 2 == 0:
                messages.append({"role": "user", "content": prompt_with_history[i]})
            else:
                messages.append(
                    {"role": "assistant", "content": prompt_with_history[i]}
                )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=temperature,
            top_p=top_p,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        return response.choices[0].message.content

    @override
    def chat_with_system_prompt(
        self,
        system_prompt: str,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        for i in range(len(prompt_with_history)):
            if i % 2 == 0:
                messages.append({"role": "user", "content": prompt_with_history[i]})
            else:
                messages.append(
                    {"role": "assistant", "content": prompt_with_history[i]}
                )

        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=temperature,
            top_p=top_p,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def query(
        self,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = 0.75,
        top_p: float = 1,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=temperature,
            top_p=top_p,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        return response.choices[0].message.content

    @override
    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = 0.75,
        top_p: float = 1,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=temperature,
            top_p=top_p,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        return response.choices[0].message.content

    @override
    def query_multimodal(
        self,
        system_prompt: Optional[str] = None,
        text_prompt: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        audio_paths: Optional[List[str]] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        # Together AI supports multiple images but not audio
        if audio_paths and len(audio_paths) > 0:
            raise UnretriableQueryException(
                "TOGETHER chat-completions does not support audio inputs yet."
            )

        if text_prompt is None and (image_paths is None or len(image_paths) == 0):
            raise ValueError(
                "At least one of text_prompt or image_paths must be given."
            )

        # Build OpenAI‑compatible message list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Compose the user content as a list for multimodal prompts
        user_content: list[dict[str, str | dict[str, str]]] = []
        if text_prompt:
            user_content.append({"type": "text", "text": text_prompt})

        if image_paths and len(image_paths) > 0:
            for image_path in image_paths:
                image_data = image_to_b64(image_path)
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    }
                )

        messages.append({"role": "user", "content": user_content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].message.content

    @override
    def valid_models(self) -> list[str]:
        return [
            "mistralai/Mistral-7B-v0.1",
            "lmsys/vicuna-7b-v1.5",
            "togethercomputer/CodeLlama-7b",
            "togethercomputer/CodeLlama-7b-Python",
            "togethercomputer/CodeLlama-7b-Instruct",
            "togethercomputer/CodeLlama-13b",
            "togethercomputer/CodeLlama-13b-Python",
            "togethercomputer/CodeLlama-13b-Instruct",
            "togethercomputer/falcon-40b",
            "togethercomputer/llama-2-7b",
            "togethercomputer/llama-2-7b-chat",
            "togethercomputer/llama-2-13b",
            "togethercomputer/llama-2-13b-chat",
            "togethercomputer/llama-2-70b",
            "togethercomputer/llama-2-70b-chat",
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        ]


class ANTHROPIC(LLM):
    """Accessing Anthropic models"""

    BASE_API_URL: str = "https://api.anthropic.com/v1"

    @override
    def query(
        self,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        url = f"{self.BASE_API_URL}/messages"
        # Define the headers
        headers = {
            "x-api-key": f"{self.api_key}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        # Define the data payload
        data = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        response = requests.post(url, headers=headers, json=data)
        json_response = response.json()
        content = json_response.get("content", [])
        if not content:
            logging.info(
                f"Failed to extract content from Anthropic API response.  Response: {response}"
            )
            return "FAIL TO QUERY: no content in response"  # Anthropic didn't return a response
        text_value = content[0].get("text", None)
        if not text_value:
            logging.info(
                f"Failed to extract text from Anthropic API response content.  Content: {content}"
            )
            return "FAIL TO QUERY: no text in response"  # Anthropic didn't return a response
        return text_value

    @override
    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        if guided_decode_json_schema is not None:
            raise NotImplementedError(
                "guided_decode_json_schema not implemented for ANTHROPIC."
            )
        url = f"{self.BASE_API_URL}/messages"
        headers = {
            "x-api-key": f"{self.api_key}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        data = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "system": system_prompt,
        }
        response = requests.post(url, headers=headers, json=data)
        json_response = response.json()
        content = json_response.get("content", [])
        if not content:
            return "FAILED TO QUERY: no content in response"  # Anthropic didn't return a response
        text_value = content[0].get("text", None)
        if not text_value:
            return "FAILED TO QUERY: no text in response"  # Anthropic didn't return a response
        return text_value

    @override
    def query_with_system_prompt_and_image(
        self,
        system_prompt: str,
        prompt: str,
        image_path: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        if guided_decode_json_schema is not None:
            raise NotImplementedError(
                "guided_decode_json_schema not implemented for ANTHROPIC."
            )
        url = f"{self.BASE_API_URL}/messages"
        headers = {
            "x-api-key": f"{self.api_key}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        data = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_to_b64(image_path),
                            },
                        },
                    ],
                }
            ],
            "system": system_prompt,
        }
        response = requests.post(url, headers=headers, json=data)
        json_response = response.json()
        content = json_response.get("content", [])
        if not content:
            return "FAILED TO QUERY: no content in response"  # Anthropic didn't return a response
        text_value = content[0].get("text", None)
        if not text_value:
            return "FAILED TO QUERY: no text in response"  # Anthropic didn't return a response
        return text_value

    @override
    def query_multimodal(
        self,
        system_prompt: Optional[str] = None,
        text_prompt: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        audio_paths: Optional[List[str]] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        # Anthropic doesn't support audio inputs yet
        if audio_paths:
            raise ValueError("Anthropic does not support audio inputs yet")

        url = f"{self.BASE_API_URL}/messages"
        headers = {
            "x-api-key": f"{self.api_key}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        content = []
        if text_prompt:
            content.append({"type": "text", "text": text_prompt})

        if image_paths:
            for image_path in image_paths:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_to_b64(image_path),
                        },
                    }
                )

        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }

        if system_prompt:
            data["system"] = system_prompt

        response = requests.post(url, headers=headers, json=data)
        json_response = response.json()
        content = json_response.get("content", [])
        if not content:
            return "FAILED TO QUERY: no content in response"  # Anthropic didn't return a response
        text_value = content[0].get("text", None)
        if not text_value:
            return "FAILED TO QUERY: no text in response"  # Anthropic didn't return a response
        return text_value

    @override
    def chat(
        self,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        raise NotImplementedError("Chat API not implemented yet.")

    @override
    def chat_with_system_prompt(
        self,
        system_prompt: str,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        raise NotImplementedError("System prompt not implemented yet.")

    @override
    def valid_models(self) -> list[str]:
        return [
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
        ]


class GOOGLEGENAI(LLM):
    """Accessing Google Gen AI (Gemini), updated with CloudModel-style logic"""

    def __init__(self, config: LLMConfig, timeout: float = 900.0, max_retries: int = 5) -> None:
        self._reasoning_models = [
            "gemini-2.5-flash", "gemini-2.5-pro"
        ]
        super().__init__(config)
        self.client = genai.Client()
        self.timeout = timeout
        self.max_retries = max_retries

    @override
    def valid_models(self) -> list[str]:
        return ["gemini-2.0-flash"] + self._reasoning_models

    @override
    def chat(self, prompt_with_history: list[str], guided_decode_json_schema: str = None,
             temperature: float = DEFAULT_TEMPERATURE, top_p: float = DEFAULT_TOP_P) -> str:
        raise NotImplementedError("Not implemented yet for GOOGLE.")

    @override
    def chat_with_system_prompt(self, system_prompt: str, prompt_with_history: list[str],
                                guided_decode_json_schema: str = None,
                                temperature: float = DEFAULT_TEMPERATURE,
                                top_p: float = DEFAULT_TOP_P) -> str:
        raise NotImplementedError("Not implemented yet for GOOGLE.")

    @override
    def query(self, prompt: str, guided_decode_json_schema: str = None,
              temperature: float = DEFAULT_TEMPERATURE, top_p: float = DEFAULT_TOP_P) -> str:
        """Send a single prompt to Gemini and return the response text."""
        try:
            print(f"Query:\n{prompt}")
            response = self.client.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction="",
                    # maxOutputTokens can be set if needed
                ),
                contents=prompt,
            )
            print(f"Response:\n{response.text}")
            return response.text or ""
        except Exception as e:
            print(f"Google GenAI request failed: {e}")
            return ""

    @override
    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        """Send a single prompt with system prompt to Gemini."""
        try:
            print(f"Query:\n{prompt}")
            response = self.client.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                ),
                contents=prompt,
            )
            print(f"Response:\n{response.text}")
            return response.text or ""
        except Exception as e:
            print(f"Google GenAI request failed: {e}")
            return ""

    @override
    def query_with_retries(
        self,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 3,
        backoff: float = 1.5,
    ) -> str:
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Query:\n{prompt}")
                response = self.client.models.generate_content(
                    model=self.model,
                    config=types.GenerateContentConfig(system_instruction=""),
                    contents=prompt,
                )
                print(f"Response:\n{response.text}")
                return response.text or ""
            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    time.sleep(backoff ** attempt)
        return ""

    @override
    def query_with_system_prompt_with_retries(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 3,
        backoff: float = 1.5,
    ) -> str:
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Query:\n{prompt}")
                response = self.client.models.generate_content(
                    model=self.model,
                    config=types.GenerateContentConfig(system_instruction=system_prompt),
                    contents=prompt,
                )
                print(f"Response:\n{response.text}")
                return response.text or ""
            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    time.sleep(backoff ** attempt)
        return ""


class DEEPSEEK:
    """Access DeepSeek Cloud API as a drop-in LLM class."""

    def __init__(
        self,
        model_name: str,
        timeout: float = 900.0
    ):
        self.model_name = model_name
        self.timeout = timeout
        self.client = openai.OpenAI(
            api_key=os.environ.get("DEEPSEEK_TOKEN"),
            base_url="https://api.deepseek.com"
        )
        print(f"\nDeepseek: Using model {self.model_name} with timeout {self.timeout}s")

    def query(
        self,
        prompt: Union[str, List[dict]],
        guided_decode_json_schema: Optional[str] = None,  # accepted for API compatibility
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        print("\nQuerying - Deepseek...\n")

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise ValueError("prompt must be str or list of dicts")

        try:
            response = self.client.with_options(timeout=self.timeout).chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )

            msg = response.choices[0].message
            reasoning_content = getattr(msg, "reasoning_content", None)
            final_content = msg.content

            combined_output = ""
            if reasoning_content:
                combined_output += f"[Reasoning]\n{reasoning_content}\n\n"
            combined_output += f"[Answer]\n{final_content}"

            return combined_output

        except Exception as e:
            print(f"DeepSeek request failed: {e}")
            return ""

    def query_with_retries(
        self,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 3,
        backoff: float = 1.5,
    ) -> str:
        print("\nQuerying With Retries - Deepseek...")
        import time

        for attempt in range(max_retries):
            response = self.query(
                prompt=prompt,
                guided_decode_json_schema=guided_decode_json_schema,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            if response:
                return response
            sleep_time = backoff * (2 ** attempt)
            print(f"Retry {attempt + 1}/{max_retries}, sleeping {sleep_time}s before retrying...")
            time.sleep(sleep_time)

        print("All retries failed.")
        return ""

    def query_with_system_prompt_with_retries(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 3,
        backoff: float = 1.5,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.query_with_retries(
            prompt=messages,
            guided_decode_json_schema=guided_decode_json_schema,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            max_retries=max_retries,
            backoff=backoff
        )

class LOCALLLM:
    """Access a local HuggingFace causal LM."""

    def __init__(
        self,
        model_name: str,
        load_in_4bit: bool = True,
        multi_gpu: bool = True,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.multi_gpu = multi_gpu
        self.device = device

        print(f"\nLOCALLLM: Loading local model {self.model_name} on device {device}")
        print(f"GPU available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        self.tokenizer = None
        self.model = None
        print(f"\nLOCALLLM: '{self.model_name}', loading with 4bit={self.load_in_4bit}, multi_gpu={self.multi_gpu}")

        bnb_config = None
        torch_dtype = torch.bfloat16
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch_dtype,
        }
        if bnb_config is not None and not "gpt-oss" in self.model_name:
            model_kwargs["quantization_config"] = bnb_config

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )

    def query(
        self,
        prompt: Union[str, list[str]],
        guided_decode_json_schema: Optional[str] = None,  # accepted for API compatibility
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
    ) -> str:
        print("\nQuerying - LOCALLLM...\n")
        
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded -> Load model before querying")

        if isinstance(prompt, str):
            prompt = [
                {"role": "user", "content": prompt},
            ]

        inputs = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )


        # Decode the generated sequence fully
        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # The inputs are tokenized, so their string representation can be regenerated
        # to identify and strip them out from the decoded output
        input_text = self.tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=True
        )

        # Remove the entire input portion from the generated output
        response = full[len(input_text):].lstrip()
        print(f"\n\n\n\n\n\nInput(THIS IS THE QUESTION):\n{input_text}\n")
        print(f"\n\n\n\n\n\nResponse(THIS IS BEING RETURNED):\n{response}\n")
        return response

    def query_with_retries(
        self,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        max_retries: int = 3,
        backoff: float = 1.5,
    ) -> str:
        print("\nQuerying With Retries - LOCALLLM...")

        """Thin retry wrapper expected by the benchmark harness."""

        return self.query(
            prompt=prompt,
            guided_decode_json_schema=guided_decode_json_schema,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
    
    def query_with_system_prompt_with_retries(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_new_tokens: int = 512,
        max_retries: int = 3,
        backoff: float = 1.5,
    ) -> str:
        print("\nQuerying With System Prompt + Retries - LOCALLLM...")

        """Retry wrapper for queries with system prompt."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        return self.query(
            prompt=messages,
            guided_decode_json_schema=guided_decode_json_schema,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )


    # def generate_batch_responses(self, prompts: Sequence[str], max_new_tokens: int = 256):
    #     return self._generate_batch_responses_generic(prompts, max_new_tokens)

    # def _generate_batch_responses_generic(self, prompts: Sequence[str], max_new_tokens: int):
    #     if self.model is None or self.tokenizer is None:
    #         raise RuntimeError("Model not loaded. Call load_model() first.")

    #     print(f"\nDefaultModel ({self.model_name}) generating batch responses...")

    #     inputs = self.tokenizer(
    #         list(prompts),
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True
    #     ).to(self.model.device)

    #     outputs = self.model.generate(
    #         input_ids=inputs["input_ids"],
    #         attention_mask=inputs["attention_mask"],
    #         max_new_tokens=max_new_tokens,
    #         do_sample=True,
    #         temperature=1.0,
    #         top_p=0.95,
    #         pad_token_id=self.tokenizer.pad_token_id,
    #         eos_token_id=self.tokenizer.eos_token_id,
    #     )
    #     print(f"\nDefaultModel ({self.model_name}) batch generation complete.\n")
    #     print(f"Outputs: {outputs}\n")
    #     return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    @override
    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        print("\nQuerying With System Prompt - LOCALLLM...")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        return self.query(
            prompt=messages,
            guided_decode_json_schema=guided_decode_json_schema,
            temperature=temperature,
            top_p=top_p,
        )


    @override
    def query_with_system_prompt_and_image(
        self,
        system_prompt: str,
        prompt: str,
        image_path: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        raise NotImplementedError("Image input not supported for LOCALLLM.")

    @override
    def query_multimodal(
        self,
        system_prompt: Optional[str] = None,
        text_prompt: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        audio_paths: Optional[List[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        raise NotImplementedError("Multimodal input not supported for LOCALLLM.")

    @override
    def chat(
        self,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        print("chat")
        raise NotImplementedError("Not implemented yet for LOCALLLM.")
        full_prompt = "\n".join(prompt_with_history)
        return self.query(full_prompt, temperature=temperature, top_p=top_p)

    @override
    def chat_with_system_prompt(
        self,
        system_prompt: str,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        print("chat with system prompt")
        raise NotImplementedError("Not implemented yet for LOCALLLM.")
        history = "\n".join(prompt_with_history)
        full_prompt = f"{system_prompt}\n{history}"
        return self.query(full_prompt, temperature=temperature, top_p=top_p)

    @override
    def valid_models(self) -> list[str]:
        return [
            "mistral-7b",
            "llama-3-8b",
            "phi-2",
            "your-custom-model-name",
        ]
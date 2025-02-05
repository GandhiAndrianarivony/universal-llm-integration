from abc import ABC, abstractmethod

import streamlit as st

from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from .settings import hf_configs


class BaseChatModel(ABC):
    @property
    @abstractmethod
    def options(self):
        pass

    @abstractmethod
    def load(sef, option: str, **kwargs):
        pass


class HuggingFaceChatModel(BaseChatModel):
    provider_name: str = "Hugging Face"

    @property
    def options(self):
        return [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        ]

    @staticmethod
    @st.cache_resource
    def _load(option, **kwargs):
        llm = HuggingFaceInferenceAPI(
            token=hf_configs.HUGGINGFACE_API_KEY,
            model_name=option,
            temperature=0.5,
            max_new_tokens=1000,
            **kwargs,
        )
        embedding = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return llm, embedding

    def load(self, option: str, **kwargs):
        return HuggingFaceChatModel._load(option, **kwargs)


class OllamaChatModel(BaseChatModel):
    provider_name: str = "ollama"

    @property
    def options(self):
        return ["qwen2:latest"]

    def load(self, option: str):
        return OllamaChatModel._load(option)

    @staticmethod
    @st.cache_resource
    def _load(option):
        host = "http://q-rag:11434"
        llm = Ollama(model=option, request_timeout=3600, base_url=host)
        embedding = OllamaEmbedding(
            model_name="nomic-embed-text:latest",
            base_url=host,
        )
        return llm, embedding


providers = [HuggingFaceChatModel.provider_name, OllamaChatModel.provider_name]

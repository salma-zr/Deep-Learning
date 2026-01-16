"""Generation module for closed-book QA."""

from src.generation.base import BaseGenerator, GenerationResult
from src.generation.openai_gen import OpenAIGenerator
from src.generation.ollama_gen import OllamaGenerator
from src.generation.openrouter_gen import OpenRouterGenerator
from src.generation.hf_gen import HuggingFaceGenerator

__all__ = [
    "BaseGenerator",
    "GenerationResult",
    "OpenAIGenerator",
    "OllamaGenerator",
    "OpenRouterGenerator",
    "HuggingFaceGenerator",
]

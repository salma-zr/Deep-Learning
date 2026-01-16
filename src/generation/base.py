"""Base classes for generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any
import time


@dataclass
class GenerationResult:
    """Result of a single generation."""
    
    prediction: str
    model: str
    prompt_id: str
    latency_ms: float
    token_usage: Optional[dict] = None
    cost_estimate: Optional[float] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "prediction": self.prediction,
            "model": self.model,
            "prompt_id": self.prompt_id,
            "latency_ms": round(self.latency_ms, 2),
            "token_usage": self.token_usage,
            "cost_estimate": self.cost_estimate,
            "error": self.error,
            "metadata": self.metadata,
        }


class BaseGenerator(ABC):
    """Base class for all generators."""
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        prompt_id: str = "default",
    ):
        """
        Initialize generator.
        
        Args:
            model: Model identifier
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens to generate
            prompt_id: Identifier for the prompt template used
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_id = prompt_id
    
    @abstractmethod
    def generate(self, prompt: str) -> GenerationResult:
        """
        Generate a response for a single prompt.
        
        Args:
            prompt: Full prompt text
            
        Returns:
            GenerationResult with prediction and metadata
        """
        pass
    
    def generate_answer(
        self,
        question: str,
        prompt_template: str,
        **format_kwargs,
    ) -> GenerationResult:
        """
        Generate an answer using a prompt template.
        
        Args:
            question: The question to answer
            prompt_template: Template string with {question} placeholder
            **format_kwargs: Additional template variables
            
        Returns:
            GenerationResult
        """
        prompt = prompt_template.format(question=question, **format_kwargs)
        return self.generate(prompt)
    
    def batch_generate(
        self,
        prompts: list[str],
        show_progress: bool = True,
    ) -> list[GenerationResult]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts
            show_progress: Show progress bar
            
        Returns:
            List of GenerationResults
        """
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(prompts, desc=f"Generating ({self.model})") if show_progress else prompts
        
        for prompt in iterator:
            result = self.generate(prompt)
            results.append(result)
        
        return results
    
    @property
    def config(self) -> dict:
        """Get generator configuration."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "prompt_id": self.prompt_id,
        }
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for token usage. Override in subclasses.
        
        Returns:
            Estimated cost in USD
        """
        return 0.0


class DummyGenerator(BaseGenerator):
    """Dummy generator for testing."""
    
    def generate(self, prompt: str) -> GenerationResult:
        """Return a dummy response."""
        return GenerationResult(
            prediction="This is a dummy response for testing purposes.",
            model=self.model,
            prompt_id=self.prompt_id,
            latency_ms=10.0,
            token_usage={"prompt_tokens": 50, "completion_tokens": 10},
            cost_estimate=0.0,
        )

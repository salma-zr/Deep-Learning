"""OpenAI API generator."""

import os
import time
from typing import Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.generation.base import BaseGenerator, GenerationResult
from src.utils.logger import get_logger
from src.utils.cache import get_cache, cache_key_for_generation

logger = get_logger(__name__)

# Pricing per 1M tokens (as of early 2024, update as needed)
OPENAI_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}


class OpenAIGenerator(BaseGenerator):
    """Generator using OpenAI API."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 256,
        prompt_id: str = "default",
        api_key: Optional[str] = None,
        use_cache: bool = True,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize OpenAI generator.
        
        Args:
            model: OpenAI model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            prompt_id: Prompt identifier
            api_key: API key (defaults to OPENAI_API_KEY env var)
            use_cache: Whether to cache responses
            system_prompt: Optional system prompt
        """
        super().__init__(model, temperature, max_tokens, prompt_id)
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not set")
        
        self.client = OpenAI(api_key=self.api_key)
        self.use_cache = use_cache
        self.cache = get_cache("openai") if use_cache else None
        self.system_prompt = system_prompt or "You are a helpful medical assistant. Answer questions concisely in one sentence."
    
    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=4, max=32),
    )
    def _call_api(self, prompt: str) -> dict:
        """Make API call with retry logic."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        return {
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "finish_reason": response.choices[0].finish_reason,
        }
    
    def generate(self, prompt: str) -> GenerationResult:
        """Generate response using OpenAI API."""
        # Check cache
        if self.use_cache:
            cache_key = cache_key_for_generation(
                self.model, prompt, "", self.temperature, self.max_tokens
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for prompt (key={cache_key[:8]})")
                return GenerationResult(**cached)
        
        # Make API call
        start_time = time.perf_counter()
        
        try:
            response = self._call_api(prompt)
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            prediction = response["content"].strip()
            usage = response["usage"]
            cost = self.estimate_cost(usage["prompt_tokens"], usage["completion_tokens"])
            
            result = GenerationResult(
                prediction=prediction,
                model=self.model,
                prompt_id=self.prompt_id,
                latency_ms=latency_ms,
                token_usage=usage,
                cost_estimate=cost,
                metadata={"finish_reason": response["finish_reason"]},
            )
            
            # Cache result
            if self.use_cache:
                self.cache.set(cache_key, result.to_dict())
            
            return result
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"OpenAI API error: {e}")
            
            return GenerationResult(
                prediction="",
                model=self.model,
                prompt_id=self.prompt_id,
                latency_ms=latency_ms,
                error=str(e),
            )
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD."""
        pricing = OPENAI_PRICING.get(self.model, {"input": 1.0, "output": 3.0})
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost

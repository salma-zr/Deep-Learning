"""OpenRouter API generator (free tier support)."""

import os
import time
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.generation.base import BaseGenerator, GenerationResult
from src.utils.logger import get_logger
from src.utils.cache import get_cache, cache_key_for_generation

logger = get_logger(__name__)

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Free models available on OpenRouter (subject to change)
FREE_MODELS = [
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-7b-it:free",
    "huggingfaceh4/zephyr-7b-beta:free",
    "openchat/openchat-7b:free",
]


class OpenRouterGenerator(BaseGenerator):
    """Generator using OpenRouter API with free tier support."""
    
    def __init__(
        self,
        model: str = "mistralai/mistral-7b-instruct:free",
        temperature: float = 0.0,
        max_tokens: int = 256,
        prompt_id: str = "default",
        api_key: Optional[str] = None,
        use_cache: bool = True,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize OpenRouter generator.
        
        Args:
            model: OpenRouter model name (use :free suffix for free models)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            prompt_id: Prompt identifier
            api_key: API key (defaults to OPENROUTER_API_KEY env var)
            use_cache: Whether to cache responses
            system_prompt: Optional system prompt
        """
        super().__init__(model, temperature, max_tokens, prompt_id)
        
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.warning("OpenRouter API key not set. Some models may not work.")
        
        self.use_cache = use_cache
        self.cache = get_cache("openrouter") if use_cache else None
        self.system_prompt = system_prompt or "You are a helpful medical assistant. Answer questions concisely in one sentence."
        
        self._client = httpx.Client(timeout=60.0)
    
    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=4, max=32),
    )
    def _call_api(self, prompt: str) -> dict:
        """Make API call with retry logic."""
        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/medical-qa-project",
            "X-Title": "Medical QA Project",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        response = self._client.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
        )
        
        response.raise_for_status()
        data = response.json()
        
        return {
            "content": data["choices"][0]["message"]["content"],
            "usage": data.get("usage", {}),
            "finish_reason": data["choices"][0].get("finish_reason"),
        }
    
    def generate(self, prompt: str) -> GenerationResult:
        """Generate response using OpenRouter API."""
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
            usage = response.get("usage")
            
            # Free models have zero cost
            cost = 0.0 if ":free" in self.model else self._estimate_cost(usage)
            
            result = GenerationResult(
                prediction=prediction,
                model=self.model,
                prompt_id=self.prompt_id,
                latency_ms=latency_ms,
                token_usage=usage,
                cost_estimate=cost,
                metadata={"finish_reason": response.get("finish_reason")},
            )
            
            # Cache result
            if self.use_cache:
                self.cache.set(cache_key, result.to_dict())
            
            return result
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"OpenRouter API error: {e}")
            
            return GenerationResult(
                prediction="",
                model=self.model,
                prompt_id=self.prompt_id,
                latency_ms=latency_ms,
                error=str(e),
            )
    
    def _estimate_cost(self, usage: Optional[dict]) -> float:
        """Estimate cost (placeholder - OpenRouter pricing varies by model)."""
        if not usage:
            return 0.0
        
        # Generic estimate - actual pricing varies
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        # Assume ~$1/1M tokens for input, ~$2/1M for output (rough average)
        return (input_tokens / 1_000_000) + (output_tokens / 1_000_000) * 2


def list_free_models() -> list[str]:
    """Return list of known free models on OpenRouter."""
    return FREE_MODELS.copy()

"""Ollama local model generator."""

import time
from typing import Optional

from src.generation.base import BaseGenerator, GenerationResult
from src.utils.logger import get_logger
from src.utils.cache import get_cache, cache_key_for_generation

logger = get_logger(__name__)


class OllamaGenerator(BaseGenerator):
    """Generator using local Ollama models."""
    
    def __init__(
        self,
        model: str = "llama3.1:8b",
        temperature: float = 0.0,
        max_tokens: int = 256,
        prompt_id: str = "default",
        host: str = "http://localhost:11434",
        use_cache: bool = True,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize Ollama generator.
        
        Args:
            model: Ollama model name (e.g., llama3.1:8b, mistral:7b)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            prompt_id: Prompt identifier
            host: Ollama server URL
            use_cache: Whether to cache responses
            system_prompt: Optional system prompt
        """
        super().__init__(model, temperature, max_tokens, prompt_id)
        
        self.host = host
        self.use_cache = use_cache
        self.cache = get_cache("ollama") if use_cache else None
        self.system_prompt = system_prompt or "You are a helpful medical assistant. Answer questions concisely in one sentence."
        
        # Import ollama here to handle cases where it's not installed
        try:
            import ollama
            self.client = ollama.Client(host=host)
            self._ollama_available = True
        except ImportError:
            logger.warning("Ollama package not installed. Install with: pip install ollama")
            self._ollama_available = False
        except Exception as e:
            logger.warning(f"Could not connect to Ollama: {e}")
            self._ollama_available = False
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        if not self._ollama_available:
            return False
        
        try:
            self.client.list()
            return True
        except Exception:
            return False
    
    def list_models(self) -> list[str]:
        """List available Ollama models."""
        if not self._ollama_available:
            return []
        
        try:
            models = self.client.list()
            return [m["name"] for m in models.get("models", [])]
        except Exception as e:
            logger.warning(f"Could not list Ollama models: {e}")
            return []
    
    def generate(self, prompt: str) -> GenerationResult:
        """Generate response using Ollama."""
        if not self._ollama_available:
            return GenerationResult(
                prediction="",
                model=self.model,
                prompt_id=self.prompt_id,
                latency_ms=0,
                error="Ollama not available",
            )
        
        # Check cache
        if self.use_cache:
            cache_key = cache_key_for_generation(
                self.model, prompt, "", self.temperature, self.max_tokens
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for prompt (key={cache_key[:8]})")
                return GenerationResult(**cached)
        
        # Make generation call
        start_time = time.perf_counter()
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            prediction = response["message"]["content"].strip()
            
            # Extract token usage if available
            usage = None
            if "prompt_eval_count" in response or "eval_count" in response:
                usage = {
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0),
                    "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
                }
            
            result = GenerationResult(
                prediction=prediction,
                model=self.model,
                prompt_id=self.prompt_id,
                latency_ms=latency_ms,
                token_usage=usage,
                cost_estimate=0.0,  # Local models are free
                metadata={
                    "total_duration": response.get("total_duration"),
                    "load_duration": response.get("load_duration"),
                },
            )
            
            # Cache result
            if self.use_cache:
                self.cache.set(cache_key, result.to_dict())
            
            return result
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Ollama error: {e}")
            
            return GenerationResult(
                prediction="",
                model=self.model,
                prompt_id=self.prompt_id,
                latency_ms=latency_ms,
                error=str(e),
            )


def check_ollama_status() -> dict:
    """Check Ollama server status and available models."""
    try:
        import ollama
        client = ollama.Client()
        models = client.list()
        return {
            "available": True,
            "models": [m["name"] for m in models.get("models", [])],
        }
    except ImportError:
        return {"available": False, "error": "Ollama package not installed"}
    except Exception as e:
        return {"available": False, "error": str(e)}

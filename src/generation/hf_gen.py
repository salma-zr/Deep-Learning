"""HuggingFace Transformers generator (local CPU/GPU)."""

import time
from typing import Optional

from src.generation.base import BaseGenerator, GenerationResult
from src.utils.logger import get_logger
from src.utils.cache import get_cache, cache_key_for_generation

logger = get_logger(__name__)


class HuggingFaceGenerator(BaseGenerator):
    """Generator using HuggingFace Transformers (CPU/GPU fallback)."""
    
    def __init__(
        self,
        model: str = "google/flan-t5-base",
        temperature: float = 0.0,
        max_tokens: int = 256,
        prompt_id: str = "default",
        device: str = "auto",
        use_cache: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize HuggingFace generator.
        
        Args:
            model: HuggingFace model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            prompt_id: Prompt identifier
            device: Device to use ('auto', 'cpu', 'cuda')
            use_cache: Whether to cache responses
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization
        """
        super().__init__(model, temperature, max_tokens, prompt_id)
        
        self.device = device
        self.use_cache = use_cache
        self.cache = get_cache("huggingface") if use_cache else None
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        self._model = None
        self._tokenizer = None
        self._loaded = False
    
    def _load_model(self):
        """Lazy load the model and tokenizer."""
        if self._loaded:
            return
        
        logger.info(f"Loading HuggingFace model: {self.model}")
        
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Determine device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            logger.info(f"Using device: {device}")
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model)
            
            # Determine model type and load
            model_kwargs = {}
            
            if device == "cuda":
                if self.load_in_4bit:
                    model_kwargs["load_in_4bit"] = True
                    model_kwargs["device_map"] = "auto"
                elif self.load_in_8bit:
                    model_kwargs["load_in_8bit"] = True
                    model_kwargs["device_map"] = "auto"
                else:
                    model_kwargs["device_map"] = "auto"
            
            # Try Seq2Seq first (T5, BART), then CausalLM (LLaMA, GPT)
            try:
                self._model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model, **model_kwargs
                )
                self._model_type = "seq2seq"
            except Exception:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model, **model_kwargs
                )
                self._model_type = "causal"
            
            if device == "cpu" and not self.load_in_8bit and not self.load_in_4bit:
                self._model = self._model.to(device)
            
            self._device = device
            self._loaded = True
            
            logger.info(f"Model loaded successfully (type: {self._model_type})")
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            raise
    
    def generate(self, prompt: str) -> GenerationResult:
        """Generate response using HuggingFace model."""
        # Check cache
        if self.use_cache:
            cache_key = cache_key_for_generation(
                self.model, prompt, "", self.temperature, self.max_tokens
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for prompt (key={cache_key[:8]})")
                return GenerationResult(**cached)
        
        # Load model if needed
        try:
            self._load_model()
        except Exception as e:
            return GenerationResult(
                prediction="",
                model=self.model,
                prompt_id=self.prompt_id,
                latency_ms=0,
                error=f"Model loading failed: {e}",
            )
        
        # Generate
        start_time = time.perf_counter()
        
        try:
            import torch
            
            # Tokenize
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            
            if self._device == "cuda" and not self.load_in_8bit and not self.load_in_4bit:
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": self.max_tokens,
                    "do_sample": self.temperature > 0,
                    "pad_token_id": self._tokenizer.eos_token_id,
                }
                
                if self.temperature > 0:
                    gen_kwargs["temperature"] = self.temperature
                
                outputs = self._model.generate(**inputs, **gen_kwargs)
            
            # Decode
            if self._model_type == "seq2seq":
                prediction = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                # For causal LM, skip the input tokens
                input_length = inputs["input_ids"].shape[1]
                prediction = self._tokenizer.decode(
                    outputs[0][input_length:], skip_special_tokens=True
                )
            
            prediction = prediction.strip()
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Token counts
            input_tokens = inputs["input_ids"].shape[1]
            output_tokens = outputs.shape[1] - (0 if self._model_type == "seq2seq" else input_tokens)
            
            result = GenerationResult(
                prediction=prediction,
                model=self.model,
                prompt_id=self.prompt_id,
                latency_ms=latency_ms,
                token_usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
                cost_estimate=0.0,  # Local models are free
                metadata={"device": self._device, "model_type": self._model_type},
            )
            
            # Cache result
            if self.use_cache:
                self.cache.set(cache_key, result.to_dict())
            
            return result
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"HuggingFace generation error: {e}")
            
            return GenerationResult(
                prediction="",
                model=self.model,
                prompt_id=self.prompt_id,
                latency_ms=latency_ms,
                error=str(e),
            )
    
    def unload(self):
        """Unload model to free memory."""
        if self._loaded:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._loaded = False
            
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Model unloaded")

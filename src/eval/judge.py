"""LLM-as-a-Judge evaluation."""

import json
import re
import time
from typing import Optional
from dataclasses import dataclass

from src.generation.prompts import load_judge_prompt
from src.utils.logger import get_logger
from src.utils.cache import get_cache

logger = get_logger(__name__)


@dataclass
class JudgeResult:
    """Result from judge evaluation."""
    
    score: int  # 0 (wrong), 1 (partial), 2 (correct)
    reasoning: str
    latency_ms: float
    error: Optional[str] = None
    raw_response: Optional[str] = None


class LLMJudge:
    """
    LLM-based judge for evaluating answer quality.
    
    Scores:
    - 2: Correct - prediction is semantically equivalent to reference
    - 1: Partial - prediction contains some correct information but is incomplete or has minor errors
    - 0: Wrong - prediction is incorrect, irrelevant, or contradicts the reference
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        backend: str = "openai",
        use_cache: bool = True,
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize the judge.
        
        Args:
            model: Model to use for judging
            backend: Backend to use (openai, ollama, etc.)
            use_cache: Whether to cache judge results
            prompt_template: Custom judge prompt template
        """
        self.model = model
        self.backend = backend
        self.use_cache = use_cache
        self.cache = get_cache("judge") if use_cache else None
        
        # Load prompt template
        if prompt_template is None:
            try:
                self.prompt_template = load_judge_prompt()
            except FileNotFoundError:
                self.prompt_template = self._default_prompt()
        else:
            self.prompt_template = prompt_template
        
        # Initialize generator
        self._generator = None
    
    def _default_prompt(self) -> str:
        """Default judge prompt."""
        return """You are evaluating medical question answering. Compare the prediction to the reference answer.

Question: {question}
Reference Answer: {reference}
Predicted Answer: {prediction}

Score the prediction:
- 2: CORRECT - semantically equivalent to reference, may use different wording
- 1: PARTIAL - contains some correct information but incomplete or has minor errors
- 0: WRONG - incorrect, irrelevant, contradicts reference, or empty

Respond with ONLY a JSON object:
{{"score": <0|1|2>, "reasoning": "<brief explanation>"}}"""
    
    def _get_generator(self):
        """Get or create the generator."""
        if self._generator is None:
            from src.generation.cli import get_generator
            
            config = {
                "backend": self.backend,
                "model": self.model,
                "temperature": 0.0,
                "max_tokens": 150,
                "prompt_id": "judge",
            }
            self._generator = get_generator(config)
        
        return self._generator
    
    def _get_cache_key(self, question: str, reference: str, prediction: str) -> str:
        """Generate cache key."""
        import hashlib
        content = f"{self.model}:{question}:{reference}:{prediction}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def _parse_response(self, response: str) -> tuple[int, str]:
        """Parse judge response to extract score and reasoning."""
        # Try to extract JSON
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                score = int(data.get("score", 0))
                reasoning = data.get("reasoning", "")
                
                # Validate score
                if score not in [0, 1, 2]:
                    score = 0
                
                return score, reasoning
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        
        # Fallback: try to find score in text
        if "2" in response or "CORRECT" in response.upper():
            return 2, "Parsed from text"
        elif "1" in response or "PARTIAL" in response.upper():
            return 1, "Parsed from text"
        else:
            return 0, "Could not parse response"
    
    def judge(
        self,
        question: str,
        reference: str,
        prediction: str,
    ) -> JudgeResult:
        """
        Judge a single prediction.
        
        Args:
            question: The original question
            reference: Reference answer
            prediction: Predicted answer
            
        Returns:
            JudgeResult with score and reasoning
        """
        # Check cache
        if self.use_cache:
            cache_key = self._get_cache_key(question, reference, prediction)
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Judge cache hit")
                return JudgeResult(**cached)
        
        # Handle empty prediction
        if not prediction or not prediction.strip():
            return JudgeResult(
                score=0,
                reasoning="Empty prediction",
                latency_ms=0,
            )
        
        # Format prompt
        prompt = self.prompt_template.format(
            question=question,
            reference=reference,
            prediction=prediction,
        )
        
        # Generate judgment
        start_time = time.perf_counter()
        
        try:
            generator = self._get_generator()
            result = generator.generate(prompt)
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            if result.error:
                return JudgeResult(
                    score=0,
                    reasoning="Generation error",
                    latency_ms=latency_ms,
                    error=result.error,
                    raw_response=result.prediction,
                )
            
            # Parse response
            score, reasoning = self._parse_response(result.prediction)
            
            judge_result = JudgeResult(
                score=score,
                reasoning=reasoning,
                latency_ms=latency_ms,
                raw_response=result.prediction,
            )
            
            # Cache result
            if self.use_cache:
                self.cache.set(cache_key, {
                    "score": score,
                    "reasoning": reasoning,
                    "latency_ms": latency_ms,
                    "raw_response": result.prediction,
                })
            
            return judge_result
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Judge error: {e}")
            
            return JudgeResult(
                score=0,
                reasoning="Error during judgment",
                latency_ms=latency_ms,
                error=str(e),
            )
    
    def batch_judge(
        self,
        examples: list[dict],  # Each dict has: question, reference, prediction
        show_progress: bool = True,
    ) -> list[JudgeResult]:
        """Judge multiple predictions."""
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(examples, desc="Judging") if show_progress else examples
        
        for ex in iterator:
            result = self.judge(
                question=ex["question"],
                reference=ex["reference"],
                prediction=ex["prediction"],
            )
            results.append(result)
        
        return results


def judge_predictions(
    predictions_file: str,
    model: str = "gpt-4o-mini",
    backend: str = "openai",
    output_file: Optional[str] = None,
) -> tuple[list[JudgeResult], dict]:
    """
    Judge predictions from a file.
    
    Args:
        predictions_file: Path to predictions JSONL
        model: Model to use for judging
        backend: Backend to use
        output_file: Optional path to save results
        
    Returns:
        Tuple of (results list, summary statistics)
    """
    from src.utils.io_utils import load_jsonl, save_jsonl
    
    # Load predictions
    predictions = load_jsonl(predictions_file)
    
    # Create judge
    judge = LLMJudge(model=model, backend=backend)
    
    # Judge all
    results = judge.batch_judge(predictions)
    
    # Add scores to predictions
    for pred, result in zip(predictions, results):
        pred["judge_score"] = result.score
        pred["judge_reasoning"] = result.reasoning
    
    # Save if output file specified
    if output_file:
        save_jsonl(predictions, output_file)
    
    # Compute statistics
    scores = [r.score for r in results if r.error is None]
    
    stats = {
        "total": len(results),
        "judged": len(scores),
        "errors": len(results) - len(scores),
        "mean_score": sum(scores) / len(scores) if scores else 0,
        "correct_pct": (scores.count(2) / len(scores) * 100) if scores else 0,
        "partial_pct": (scores.count(1) / len(scores) * 100) if scores else 0,
        "wrong_pct": (scores.count(0) / len(scores) * 100) if scores else 0,
    }
    
    return results, stats


def verify_judge_consistency(
    examples: list[dict],
    judge: LLMJudge,
    n_samples: int = 50,
    n_repeats: int = 2,
) -> dict:
    """
    Verify judge consistency by re-judging samples.
    
    Args:
        examples: List of examples to sample from
        judge: LLMJudge instance
        n_samples: Number of samples to verify
        n_repeats: Number of times to judge each sample
        
    Returns:
        Dictionary with consistency statistics
    """
    import random
    
    # Sample examples
    samples = random.sample(examples, min(n_samples, len(examples)))
    
    # Judge each sample multiple times
    disagreements = 0
    total = 0
    
    for sample in samples:
        scores = []
        for _ in range(n_repeats):
            # Disable cache for verification
            judge.use_cache = False
            result = judge.judge(
                question=sample["question"],
                reference=sample["reference"],
                prediction=sample["prediction"],
            )
            scores.append(result.score)
        
        # Check consistency
        if len(set(scores)) > 1:
            disagreements += 1
        total += 1
    
    # Re-enable cache
    judge.use_cache = True
    
    return {
        "samples_checked": total,
        "disagreements": disagreements,
        "consistency_rate": (total - disagreements) / total if total else 1.0,
    }

"""Evaluation metrics for Medical QA."""

from typing import Optional
import statistics

from src.utils.logger import get_logger
from src.utils.normalize import get_answer_stats

logger = get_logger(__name__)


def compute_rouge(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """
    Compute ROUGE scores.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, ROUGE-L F1 scores
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.error("rouge-score not installed. Install with: pip install rouge-score")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = {
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
    }
    
    for pred, ref in zip(predictions, references):
        # Handle empty predictions
        if not pred or not pred.strip():
            pred = "no answer"
        if not ref or not ref.strip():
            ref = "no answer"
        
        result = scorer.score(ref, pred)
        
        scores["rouge1"].append(result["rouge1"].fmeasure)
        scores["rouge2"].append(result["rouge2"].fmeasure)
        scores["rougeL"].append(result["rougeL"].fmeasure)
    
    return {
        "rouge1": statistics.mean(scores["rouge1"]),
        "rouge2": statistics.mean(scores["rouge2"]),
        "rougeL": statistics.mean(scores["rougeL"]),
    }


def compute_per_example_rouge(
    predictions: list[str],
    references: list[str],
) -> list[dict[str, float]]:
    """
    Compute per-example ROUGE scores.

    Returns a list of dicts with rouge1/rouge2/rougeL for each example.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.error("rouge-score not installed. Install with: pip install rouge-score")
        return [{"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0} for _ in predictions]

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    per_example = []

    for pred, ref in zip(predictions, references):
        if not pred or not pred.strip():
            pred = "no answer"
        if not ref or not ref.strip():
            ref = "no answer"

        result = scorer.score(ref, pred)
        per_example.append({
            "rouge1": result["rouge1"].fmeasure,
            "rouge2": result["rouge2"].fmeasure,
            "rougeL": result["rougeL"].fmeasure,
        })

    return per_example


def annotate_with_rouge(
    predictions: list[dict],
) -> list[dict]:
    """Annotate prediction records with per-example ROUGE scores."""
    preds = [p.get("prediction", "") for p in predictions]
    refs = [p.get("reference", "") for p in predictions]
    per_example = compute_per_example_rouge(preds, refs)

    for record, scores in zip(predictions, per_example):
        record["rouge_1"] = scores["rouge1"]
        record["rouge_2"] = scores["rouge2"]
        record["rouge_l"] = scores["rougeL"]

    return predictions


def compute_bleu(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """
    Compute BLEU score using sacrebleu.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers (each is a single reference)
        
    Returns:
        Dictionary with BLEU score
    """
    try:
        import sacrebleu
    except ImportError:
        logger.error("sacrebleu not installed. Install with: pip install sacrebleu")
        return {"bleu": 0.0}
    
    # sacrebleu expects references as list of lists
    refs = [[ref if ref and ref.strip() else "no answer"] for ref in references]
    preds = [pred if pred and pred.strip() else "no answer" for pred in predictions]
    
    # Compute corpus-level BLEU
    bleu = sacrebleu.corpus_bleu(preds, list(zip(*refs)))
    
    return {"bleu": bleu.score / 100}  # Normalize to 0-1 range


def compute_exact_match(
    predictions: list[str],
    references: list[str],
    normalize: bool = True,
) -> dict[str, float]:
    """
    Compute exact match score.
    
    Note: Exact match is generally NOT recommended for open-ended QA
    as it's too strict. Included for completeness.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        normalize: Whether to normalize text before comparison
        
    Returns:
        Dictionary with exact match score
    """
    matches = 0
    
    for pred, ref in zip(predictions, references):
        if normalize:
            pred = pred.lower().strip() if pred else ""
            ref = ref.lower().strip() if ref else ""
        
        if pred == ref:
            matches += 1
    
    return {"exact_match": matches / len(predictions) if predictions else 0.0}


def compute_format_stats(
    predictions: list[str],
) -> dict[str, float]:
    """
    Compute format statistics for predictions.
    
    Args:
        predictions: List of predicted answers
        
    Returns:
        Dictionary with format statistics
    """
    stats = get_answer_stats(predictions)
    
    return {
        "avg_length_chars": stats["avg_length"],
        "avg_length_words": stats["avg_words"],
        "multi_sentence_pct": stats["multi_sentence_pct"],
        "empty_pct": stats["empty_pct"],
        "insufficient_info_pct": stats["insufficient_pct"],
    }


def compute_latency_stats(
    latencies_ms: list[float],
) -> dict[str, float]:
    """
    Compute latency statistics.
    
    Args:
        latencies_ms: List of latencies in milliseconds
        
    Returns:
        Dictionary with latency statistics
    """
    if not latencies_ms:
        return {
            "latency_mean_ms": 0.0,
            "latency_median_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_min_ms": 0.0,
            "latency_max_ms": 0.0,
        }
    
    sorted_latencies = sorted(latencies_ms)
    p95_idx = int(len(sorted_latencies) * 0.95)
    
    return {
        "latency_mean_ms": statistics.mean(latencies_ms),
        "latency_median_ms": statistics.median(latencies_ms),
        "latency_p95_ms": sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)],
        "latency_min_ms": min(latencies_ms),
        "latency_max_ms": max(latencies_ms),
    }


def compute_all_metrics(
    predictions: list[str],
    references: list[str],
    latencies_ms: Optional[list[float]] = None,
    judge_scores: Optional[list[int]] = None,
    cost_estimates: Optional[list[float]] = None,
) -> dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        latencies_ms: Optional list of latencies
        judge_scores: Optional list of judge scores (0, 1, or 2)
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # ROUGE
    rouge_scores = compute_rouge(predictions, references)
    metrics.update(rouge_scores)
    
    # BLEU
    bleu_scores = compute_bleu(predictions, references)
    metrics.update(bleu_scores)
    
    # Format stats
    format_stats = compute_format_stats(predictions)
    metrics.update(format_stats)
    
    # Latency stats
    if latencies_ms:
        latency_stats = compute_latency_stats(latencies_ms)
        metrics.update(latency_stats)
    
    # Judge scores
    if judge_scores:
        valid_scores = [s for s in judge_scores if s is not None]
        if valid_scores:
            metrics["judge_mean"] = statistics.mean(valid_scores)
            metrics["judge_correct_pct"] = (valid_scores.count(2) / len(valid_scores)) * 100
            metrics["judge_partial_pct"] = (valid_scores.count(1) / len(valid_scores)) * 100
            metrics["judge_wrong_pct"] = (valid_scores.count(0) / len(valid_scores)) * 100

    # Cost estimates
    if cost_estimates:
        valid_costs = [c for c in cost_estimates if c is not None]
        if valid_costs:
            metrics["cost_total_usd"] = sum(valid_costs)
            metrics["cost_per_example_usd"] = sum(valid_costs) / len(valid_costs)
    
    # Add count
    metrics["n_examples"] = len(predictions)
    
    return metrics


def metrics_to_csv_row(
    experiment_name: str,
    metrics: dict[str, float],
) -> dict[str, any]:
    """
    Convert metrics to a CSV row format.
    
    Args:
        experiment_name: Name of the experiment
        metrics: Dictionary of metrics
        
    Returns:
        Dictionary suitable for CSV writing
    """
    row = {"experiment": experiment_name}
    
    # Round floats for readability
    for key, value in metrics.items():
        if isinstance(value, float):
            row[key] = round(value, 4)
        else:
            row[key] = value
    
    return row

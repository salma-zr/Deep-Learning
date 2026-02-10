"""Download and prepare the Medical Flashcards dataset."""

import hashlib
from pathlib import Path
from typing import Optional

from datasets import load_dataset, Dataset
import pandas as pd

from src.utils.io_utils import save_jsonl, ensure_dir, get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Dataset configuration
DATASET_NAME = "medalpaca/medical_meadow_medical_flashcards"
EXPECTED_COLUMNS = {"input", "output"}


def download_dataset(
    output_dir: Optional[str | Path] = None,
    force: bool = False,
) -> Path:
    """
    Download the Medical Flashcards dataset from HuggingFace.
    
    Args:
        output_dir: Directory to save the dataset. Defaults to data/raw/
        force: If True, re-download even if file exists
        
    Returns:
        Path to the downloaded JSONL file
    """
    if output_dir is None:
        output_dir = get_project_root() / "data" / "raw"
    
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    output_file = output_dir / "medical_flashcards.jsonl"
    
    if output_file.exists() and not force:
        logger.info(f"Dataset already exists at {output_file}")
        return output_file
    
    logger.info(f"Downloading dataset: {DATASET_NAME}")
    
    try:
        # Load from HuggingFace
        dataset = load_dataset(DATASET_NAME, split="train")
        
        # Validate columns
        missing_cols = EXPECTED_COLUMNS - set(dataset.column_names)
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
        
        logger.info(f"Downloaded {len(dataset)} examples")
        
        # Convert to list of dicts with standardized format
        records = []
        for idx, example in enumerate(dataset):
            record = {
                "id": f"flashcard_{idx:05d}",
                "question": example["input"].strip(),
                "answer": example["output"].strip(),
            }
            records.append(record)
        
        # Save as JSONL
        save_jsonl(records, output_file)
        logger.info(f"Saved dataset to {output_file}")
        
        # Log statistics
        questions = [r["question"] for r in records]
        answers = [r["answer"] for r in records]
        
        logger.info(f"Question length: mean={sum(len(q) for q in questions)/len(questions):.0f} chars")
        logger.info(f"Answer length: mean={sum(len(a) for a in answers)/len(answers):.0f} chars")
        
        return output_file
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise


def load_raw_dataset(path: Optional[str | Path] = None) -> list[dict]:
    """
    Load the raw dataset from disk.
    
    Args:
        path: Path to the JSONL file. Defaults to data/raw/medical_flashcards.jsonl
        
    Returns:
        List of records with {id, question, answer}
    """
    if path is None:
        path = get_project_root() / "data" / "raw" / "medical_flashcards.jsonl"
    
    path = Path(path)
    
    if not path.exists():
        logger.info("Dataset not found, downloading...")
        download_dataset()
    
    from src.utils.io_utils import load_jsonl
    return load_jsonl(path)


def get_dataset_info(records: list[dict]) -> dict:
    """Get statistics about the dataset."""
    questions = [r["question"] for r in records]
    answers = [r["answer"] for r in records]
    
    return {
        "num_examples": len(records),
        "question_stats": {
            "mean_length": sum(len(q) for q in questions) / len(questions),
            "min_length": min(len(q) for q in questions),
            "max_length": max(len(q) for q in questions),
            "mean_words": sum(len(q.split()) for q in questions) / len(questions),
        },
        "answer_stats": {
            "mean_length": sum(len(a) for a in answers) / len(answers),
            "min_length": min(len(a) for a in answers),
            "max_length": max(len(a) for a in answers),
            "mean_words": sum(len(a.split()) for a in answers) / len(answers),
        },
    }


def compute_dataset_hash(records: list[dict]) -> str:
    """Compute a hash of the dataset for reproducibility tracking."""
    content = "".join(f"{r['id']}:{r['question']}:{r['answer']}" for r in records)
    return hashlib.sha256(content.encode()).hexdigest()[:16]

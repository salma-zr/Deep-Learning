"""Create and manage train/dev/test splits."""

import random
from pathlib import Path
from typing import Optional

from src.utils.io_utils import (
    load_jsonl,
    save_jsonl,
    save_yaml,
    load_yaml,
    ensure_dir,
    get_project_root,
)
from src.utils.logger import get_logger
from src.data.download import load_raw_dataset, compute_dataset_hash

logger = get_logger(__name__)

# Default split configuration
DEFAULT_SPLIT_CONFIG = {
    "train_ratio": 0.80,
    "dev_ratio": 0.10,
    "test_ratio": 0.10,
    "seed": 42,
    "tiny_test_size": 200,  # Small subset for quick iterations
}


def create_splits(
    output_dir: Optional[str | Path] = None,
    train_ratio: float = 0.80,
    dev_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
    tiny_test_size: int = 200,
    force: bool = False,
) -> dict[str, list[dict]]:
    """
    Create reproducible train/dev/test splits.
    
    Args:
        output_dir: Directory to save splits. Defaults to data/splits/
        train_ratio: Proportion for training
        dev_ratio: Proportion for development/validation
        test_ratio: Proportion for testing
        seed: Random seed for reproducibility
        tiny_test_size: Size of tiny test set for quick iterations
        force: If True, recreate even if splits exist
        
    Returns:
        Dictionary with train, dev, test, tiny_test splits
    """
    if output_dir is None:
        output_dir = get_project_root() / "data" / "splits"
    
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    # Check if splits already exist
    splits_exist = all(
        (output_dir / f"{split}.jsonl").exists()
        for split in ["train", "dev", "test", "tiny_test"]
    )
    
    if splits_exist and not force:
        logger.info("Splits already exist, loading from disk")
        return load_splits(output_dir)
    
    # Validate ratios
    total_ratio = train_ratio + dev_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    # Load raw dataset
    records = load_raw_dataset()
    n_total = len(records)
    
    logger.info(f"Creating splits from {n_total} examples (seed={seed})")
    
    # Shuffle with seed
    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)
    
    # Calculate split sizes
    n_train = int(n_total * train_ratio)
    n_dev = int(n_total * dev_ratio)
    n_test = n_total - n_train - n_dev  # Remainder goes to test
    
    # Create splits
    train_data = shuffled[:n_train]
    dev_data = shuffled[n_train:n_train + n_dev]
    test_data = shuffled[n_train + n_dev:]
    
    # Create tiny test (random sample from test)
    tiny_test_data = random.sample(test_data, min(tiny_test_size, len(test_data)))
    
    splits = {
        "train": train_data,
        "dev": dev_data,
        "test": test_data,
        "tiny_test": tiny_test_data,
    }
    
    # Save splits
    for split_name, split_data in splits.items():
        save_jsonl(split_data, output_dir / f"{split_name}.jsonl")
    
    # Save metadata
    metadata = {
        "seed": seed,
        "split_ratios": {
            "train": train_ratio,
            "dev": dev_ratio,
            "test": test_ratio,
        },
        "split_sizes": {name: len(data) for name, data in splits.items()},
        "total_examples": n_total,
        "dataset_hash": compute_dataset_hash(records),
        "tiny_test_size": tiny_test_size,
    }
    save_yaml(metadata, output_dir / "metadata.yaml")
    
    logger.info(f"Created splits: train={n_train}, dev={n_dev}, test={n_test}, tiny_test={len(tiny_test_data)}")
    
    return splits


def load_splits(
    splits_dir: Optional[str | Path] = None,
) -> dict[str, list[dict]]:
    """
    Load existing splits from disk.
    
    Args:
        splits_dir: Directory containing split files
        
    Returns:
        Dictionary with train, dev, test, tiny_test splits
    """
    if splits_dir is None:
        splits_dir = get_project_root() / "data" / "splits"
    
    splits_dir = Path(splits_dir)
    
    if not splits_dir.exists():
        raise FileNotFoundError(f"Splits directory not found: {splits_dir}")
    
    splits = {}
    for split_name in ["train", "dev", "test", "tiny_test"]:
        split_file = splits_dir / f"{split_name}.jsonl"
        if split_file.exists():
            splits[split_name] = load_jsonl(split_file)
        else:
            logger.warning(f"Split file not found: {split_file}")
    
    return splits


def load_split(
    split_name: str,
    splits_dir: Optional[str | Path] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """
    Load a single split.
    
    Args:
        split_name: Name of split (train, dev, test, tiny_test)
        splits_dir: Directory containing split files
        limit: Optional limit on number of examples
        
    Returns:
        List of examples
    """
    if splits_dir is None:
        splits_dir = get_project_root() / "data" / "splits"
    
    splits_dir = Path(splits_dir)
    split_file = splits_dir / f"{split_name}.jsonl"
    
    if not split_file.exists():
        logger.info(f"Split {split_name} not found, creating splits...")
        create_splits()
    
    data = load_jsonl(split_file)
    
    if limit is not None:
        data = data[:limit]
    
    return data


def get_splits_metadata(splits_dir: Optional[str | Path] = None) -> dict:
    """Load splits metadata."""
    if splits_dir is None:
        splits_dir = get_project_root() / "data" / "splits"
    
    metadata_file = Path(splits_dir) / "metadata.yaml"
    
    if not metadata_file.exists():
        return {}
    
    return load_yaml(metadata_file)

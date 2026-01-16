"""Data preparation for fine-tuning."""

from pathlib import Path
from typing import Optional

from src.utils.io_utils import load_jsonl, save_jsonl, ensure_dir, get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


def format_instruction(question: str, answer: str, prompt_style: str = "simple") -> str:
    """
    Format a Q&A pair into instruction format for training.
    
    Args:
        question: The medical question
        answer: The reference answer
        prompt_style: Style of prompt formatting
        
    Returns:
        Formatted instruction string
    """
    if prompt_style == "simple":
        return f"Question: {question}\nAnswer: {answer}"
    
    elif prompt_style == "alpaca":
        return f"""### Instruction:
Answer the following medical question in one concise sentence.

### Input:
{question}

### Response:
{answer}"""
    
    elif prompt_style == "chat":
        return f"""<|user|>
{question}
<|assistant|>
{answer}"""
    
    elif prompt_style == "flashcard":
        return f"""Medical Flashcard Question: {question}

Correct Answer: {answer}"""
    
    else:
        raise ValueError(f"Unknown prompt style: {prompt_style}")


def prepare_training_data(
    split_name: str = "train",
    output_dir: Optional[str | Path] = None,
    max_examples: Optional[int] = None,
    prompt_style: str = "alpaca",
    add_eos: bool = True,
) -> Path:
    """
    Prepare training data for fine-tuning.
    
    Args:
        split_name: Name of split to use (train, dev)
        output_dir: Output directory for prepared data
        max_examples: Maximum number of examples (for ablations)
        prompt_style: Style of instruction formatting
        add_eos: Whether to add EOS token placeholder
        
    Returns:
        Path to the prepared data file
    """
    if output_dir is None:
        output_dir = get_project_root() / "data" / "finetune"
    
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    # Load split
    from src.data.split import load_split
    data = load_split(split_name)
    
    # Limit if specified
    if max_examples is not None:
        data = data[:max_examples]
    
    logger.info(f"Preparing {len(data)} examples for fine-tuning (style: {prompt_style})")
    
    # Format data
    prepared = []
    for example in data:
        text = format_instruction(
            example["question"],
            example["answer"],
            prompt_style=prompt_style,
        )
        
        if add_eos:
            text = text + "</s>"
        
        prepared.append({
            "id": example["id"],
            "text": text,
            "question": example["question"],
            "answer": example["answer"],
        })
    
    # Save
    suffix = f"_{max_examples}" if max_examples else ""
    output_file = output_dir / f"{split_name}_{prompt_style}{suffix}.jsonl"
    save_jsonl(prepared, output_file)
    
    logger.info(f"Saved prepared data to {output_file}")
    
    return output_file


def create_train_sizes(
    sizes: list[int] = [1000, 5000, 20000],
    prompt_style: str = "alpaca",
) -> dict[int, Path]:
    """
    Create training files of different sizes for ablation studies.
    
    Args:
        sizes: List of training set sizes
        prompt_style: Style of instruction formatting
        
    Returns:
        Dictionary mapping size to file path
    """
    result = {}
    
    for size in sizes:
        path = prepare_training_data(
            split_name="train",
            max_examples=size,
            prompt_style=prompt_style,
        )
        result[size] = path
    
    return result


def load_training_dataset(
    path: str | Path,
    tokenizer,
    max_length: int = 512,
):
    """
    Load training data as a HuggingFace Dataset.
    
    Args:
        path: Path to prepared JSONL file
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        
    Returns:
        HuggingFace Dataset ready for training
    """
    from datasets import Dataset
    
    data = load_jsonl(path)
    
    # Create dataset
    dataset = Dataset.from_list([{"text": d["text"]} for d in data])
    
    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    
    return tokenized

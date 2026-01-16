import argparse
from typing import Any, Dict

from datasets import load_dataset

from src.utils import ensure_dir, log_config, log_versions, setup_logger, write_jsonl


QUESTION_KEYS = ("question", "instruction", "input", "prompt")
ANSWER_KEYS = ("answer", "output", "response", "completion")


def normalize_example(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    question = ""
    for key in QUESTION_KEYS:
        if example.get(key):
            question = str(example[key]).strip()
            break
    answer = ""
    for key in ANSWER_KEYS:
        if example.get(key):
            answer = str(example[key]).strip()
            break
    return {
        "id": idx,
        "question": question,
        "answer": answer,
        "raw": example,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Medical Meadow dataset.")
    parser.add_argument(
        "--dataset",
        default="medalpaca/medical_meadow_medical_flashcards",
        help="Hugging Face dataset name",
    )
    parser.add_argument("--split", default="train", help="Dataset split to fetch")
    parser.add_argument("--output", default="data/raw/flashcards.jsonl", help="Output jsonl")
    args = parser.parse_args()

    logger = setup_logger("download_dataset")
    log_versions(logger)
    log_config(logger, {"dataset": args.dataset, "split": args.split, "output": args.output})
    logger.info("Loading dataset %s split=%s", args.dataset, args.split)
    dataset = load_dataset(args.dataset, split=args.split)
    logger.info("Dataset rows: %s", len(dataset))

    ensure_dir("data/raw")
    rows = [normalize_example(example, idx) for idx, example in enumerate(dataset)]
    write_jsonl(rows, args.output)
    logger.info("Saved %s rows to %s", len(rows), args.output)


if __name__ == "__main__":
    main()

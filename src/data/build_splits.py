import argparse
import random
from typing import Dict, List

from src.utils import ensure_dir, log_config, log_versions, read_jsonl, setup_logger, write_json, write_jsonl


def split_dataset(rows: List[dict], seed: int) -> Dict[str, List[dict]]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(0.8 * n)
    n_dev = int(0.1 * n)
    train = rows[:n_train]
    dev = rows[n_train : n_train + n_dev]
    test = rows[n_train + n_dev :]
    return {"train": train, "dev": dev, "test": test}


def filter_rows(rows: List[dict]) -> List[dict]:
    filtered = []
    for row in rows:
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            continue
        filtered.append({"id": row.get("id"), "question": question, "answer": answer})
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/dev/test splits.")
    parser.add_argument("--input", default="data/raw/flashcards.jsonl", help="Input jsonl")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tiny-test-size", type=int, default=200)
    parser.add_argument("--output-dir", default="data/splits")
    args = parser.parse_args()

    logger = setup_logger("build_splits")
    log_versions(logger)
    log_config(
        logger,
        {
            "input": args.input,
            "seed": args.seed,
            "tiny_test_size": args.tiny_test_size,
            "output_dir": args.output_dir,
        },
    )
    rows = read_jsonl(args.input)
    logger.info("Loaded %s rows", len(rows))
    rows = filter_rows(rows)
    logger.info("Filtered to %s rows with question+answer", len(rows))

    splits = split_dataset(rows, args.seed)
    ensure_dir(args.output_dir)
    for name, subset in splits.items():
        write_jsonl(subset, f"{args.output_dir}/{name}.jsonl")
        logger.info("Saved %s rows to %s/%s.jsonl", len(subset), args.output_dir, name)

    tiny_size = min(args.tiny_test_size, len(splits["test"]))
    tiny_test = splits["test"][:tiny_size]
    write_jsonl(tiny_test, f"{args.output_dir}/tiny_test.jsonl")
    logger.info("Saved %s rows to %s/tiny_test.jsonl", len(tiny_test), args.output_dir)

    metadata = {
        "seed": args.seed,
        "sizes": {k: len(v) for k, v in splits.items()},
        "tiny_test_size": tiny_size,
    }
    write_json(metadata, f"{args.output_dir}/metadata.json")
    logger.info("Saved metadata to %s/metadata.json", args.output_dir)


if __name__ == "__main__":
    main()

import argparse
from typing import List

from src.utils import load_config, log_config, log_versions, read_jsonl, setup_logger, write_jsonl


def build_seq2seq(rows: List[dict], prompt_template: str) -> List[dict]:
    formatted = []
    for row in rows:
        prompt = prompt_template.format(question=row["question"])
        formatted.append({"input": prompt, "output": row["answer"]})
    return formatted


def build_causal(rows: List[dict], prompt_template: str) -> List[dict]:
    formatted = []
    for row in rows:
        prompt = prompt_template.format(question=row["question"])
        formatted.append({"text": f"{prompt}\n{row['answer']}"})
    return formatted


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare data for fine-tuning.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_name = cfg["exp_name"]
    logger = setup_logger(f"{exp_name}_prep", cfg.get("logging", {}).get("log_path"))
    log_versions(logger)
    log_config(logger, cfg)

    rows = read_jsonl(cfg["data"]["train_path"])
    limit = cfg["data"].get("train_limit", cfg["data"].get("limit"))
    if limit:
        rows = rows[: int(limit)]

    fmt = cfg["finetune"].get("format", "seq2seq")
    prompt_template = cfg["finetune"]["prompt_template"]
    if fmt == "seq2seq":
        data = build_seq2seq(rows, prompt_template)
    else:
        data = build_causal(rows, prompt_template)

    out_path = cfg["finetune"]["prepared_path"]
    write_jsonl(data, out_path)
    logger.info("Saved %s finetune rows to %s", len(data), out_path)


if __name__ == "__main__":
    main()

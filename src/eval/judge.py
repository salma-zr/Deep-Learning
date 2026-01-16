import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from src.utils import (
    chat_completion,
    estimate_cost,
    load_config,
    read_jsonl,
    read_text,
    log_config,
    log_versions,
    setup_logger,
    set_seed,
    should_skip,
    timed,
    write_jsonl,
)


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(text: str) -> Dict[str, Any]:
    match = JSON_RE.search(text)
    if not match:
        return {"score": None, "rationale": "No JSON found.", "raw": text}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"score": None, "rationale": "Invalid JSON.", "raw": text}


def build_messages(prompt_template: str, question: str, reference: str, prediction: str) -> List[Dict[str, str]]:
    user = prompt_template.format(question=question, reference=reference, prediction=prediction)
    return [{"role": "user", "content": user}]


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-a-judge scoring.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--preds", help="Override path to predictions jsonl")
    parser.add_argument("--output", help="Override path to judge output jsonl")
    parser.add_argument("--exp-name", help="Override experiment name")
    parser.add_argument("--limit", type=int, help="Limit number of rows judged")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_name = args.exp_name or cfg.get("exp_name", "judge")
    logger = setup_logger(f"{exp_name}_judge", cfg.get("logging", {}).get("log_path"))
    log_versions(logger)
    log_config(logger, cfg)
    set_seed(cfg.get("seed", 42))

    output_path = Path(args.output or cfg["output"]["judge_path"])
    use_cache = cfg["output"].get("cache", True)
    if should_skip(output_path, use_cache):
        logger.info("Skipping %s (output exists).", output_path)
        return

    preds = read_jsonl(args.preds or cfg["output"]["pred_path"])
    limit = args.limit if args.limit is not None else cfg["data"].get("limit")
    if limit:
        preds = preds[: int(limit)]

    prompt_template = read_text(cfg["prompt"]["path"])

    from src.utils import get_openai_client

    base_url = cfg["model"].get("base_url")
    api_env = cfg["model"].get("api_key_env", "OPENAI_API_KEY")
    client = get_openai_client(base_url=base_url, api_key_env=api_env)

    model_name = cfg["model"]["name"]
    temperature = cfg["model"].get("temperature", 0.0)
    max_tokens = cfg["model"].get("max_tokens", 128)
    pricing = cfg["model"].get("pricing")

    results: List[Dict[str, Any]] = []
    for row in preds:
        with timed() as tinfo:
            response = chat_completion(
                client=client,
                model=model_name,
                messages=build_messages(prompt_template, row["question"], row["reference"], row["prediction"]),
                temperature=temperature,
                max_tokens=max_tokens,
                extra=cfg["model"].get("extra"),
            )
        parsed = parse_json(response["content"])
        usage = response.get("usage", {})

        results.append(
            {
                "id": row["id"],
                "score": parsed.get("score"),
                "rationale": parsed.get("rationale"),
                "raw": parsed.get("raw"),
                "meta": {
                    "model": model_name,
                    "latency_ms": tinfo["elapsed_ms"],
                    "token_usage": usage,
                    "cost_estimate": estimate_cost(usage, pricing),
                },
            }
        )

    write_jsonl(results, output_path)
    logger.info("Saved %s judge rows to %s", len(results), output_path)


if __name__ == "__main__":
    main()

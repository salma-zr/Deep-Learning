import argparse
from pathlib import Path

from src.rag.retriever import retrieve_duckduckgo, retrieve_wikipedia
from src.utils import load_config, log_config, log_versions, read_jsonl, setup_logger, should_skip, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval cache for RAG.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_name = cfg["exp_name"]
    logger = setup_logger(f"{exp_name}_index", cfg.get("logging", {}).get("log_path"))
    log_versions(logger)
    log_config(logger, cfg)

    data_rows = read_jsonl(cfg["data"]["split_path"])
    limit = cfg["data"].get("limit")
    if limit:
        data_rows = data_rows[: int(limit)]

    retrieval = cfg["retrieval"]
    cache_path = Path(retrieval["cache_path"])
    use_cache = retrieval.get("cache", True)
    if should_skip(cache_path, use_cache):
        logger.info("Skipping retrieval cache %s (exists).", cache_path)
        return

    provider = retrieval.get("provider", "wikipedia")
    search_top_k = retrieval.get("search_top_k", 3)
    lang = retrieval.get("lang", "en")

    rows = []
    for row in data_rows:
        question = row["question"]
        if provider == "wikipedia":
            docs = retrieve_wikipedia(question, top_k=search_top_k, lang=lang)
        elif provider == "duckduckgo":
            docs = retrieve_duckduckgo(question, top_k=search_top_k)
        else:
            raise ValueError(f"Unknown retrieval provider: {provider}")

        rows.append({"id": row["id"], "question": question, "docs": docs})

    write_jsonl(rows, cache_path)
    logger.info("Saved retrieval cache with %s rows to %s", len(rows), cache_path)


if __name__ == "__main__":
    main()

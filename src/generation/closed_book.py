import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils import (
    chat_completion,
    estimate_cost,
    load_config,
    normalize_reference,
    postprocess_one_sentence,
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


def load_prompts(prompt_path: str) -> Dict[str, Dict[str, str]]:
    data = read_text(prompt_path)
    payload = json.loads(data) if prompt_path.endswith(".json") else None
    if payload:
        prompts = payload.get("prompts", [])
    else:
        import yaml

        payload = yaml.safe_load(data)
        prompts = payload.get("prompts", [])
    return {prompt["id"]: prompt for prompt in prompts}


def build_messages(prompt: Dict[str, str], question: str, context: Optional[str] = None) -> List[Dict[str, str]]:
    user_template = prompt.get("user", "")
    user = user_template.format(question=question, context=context or "")
    messages = []
    system = prompt.get("system", "")
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


def build_prompt_text(prompt: Dict[str, str], question: str, context: Optional[str] = None) -> str:
    system = prompt.get("system", "")
    user_template = prompt.get("user", "")
    user = user_template.format(question=question, context=context or "")
    if system:
        return f"{system}\n\n{user}"
    return user


def load_hf_model(model_name: str, model_type: str, adapter_path: Optional[str]) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    return tokenizer, model


def generate_hf(
    tokenizer: Any,
    model: Any,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if decoded.startswith(prompt_text):
        decoded = decoded[len(prompt_text) :]
    return decoded.strip()


def generate_ollama(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    base_url: str,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "stream": False,
    }
    response = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return {
        "content": data.get("message", {}).get("content", ""),
        "usage": {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
        },
    }


def load_existing(path: Path) -> Dict[Any, Dict[str, Any]]:
    if not path.exists():
        return {}
    existing = {}
    for row in read_jsonl(path):
        existing[row.get("id")] = row
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-book and prompt-based generation.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_name = cfg["exp_name"]
    logger = setup_logger(exp_name, cfg.get("logging", {}).get("log_path"))
    log_versions(logger)
    log_config(logger, cfg)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    output_path = Path(cfg["output"]["pred_path"])
    use_cache = cfg["output"].get("cache", True)
    if should_skip(output_path, use_cache):
        logger.info("Skipping %s (output exists).", output_path)
        return

    data_rows = read_jsonl(cfg["data"]["split_path"])
    limit = cfg["data"].get("limit")
    if limit:
        data_rows = data_rows[: int(limit)]

    prompt_cfg = load_prompts(cfg["prompt"]["path"])
    prompt = prompt_cfg[cfg["prompt"]["prompt_id"]]

    provider = cfg["model"]["provider"]
    model_name = cfg["model"]["name"]
    temperature = cfg["model"].get("temperature", 0.2)
    max_tokens = cfg["model"].get("max_tokens", 64)
    pricing = cfg["model"].get("pricing")
    postprocess = cfg["prompt"].get("postprocess_one_sentence", False)

    existing = load_existing(output_path) if use_cache else {}
    results: List[Dict[str, Any]] = list(existing.values())

    hf_bundle = None
    client = None
    if provider in {"openai", "openrouter"}:
        from src.utils import get_openai_client

        base_url = cfg["model"].get("base_url")
        api_env = cfg["model"].get("api_key_env", "OPENAI_API_KEY")
        client = get_openai_client(base_url=base_url, api_key_env=api_env)
    elif provider == "hf":
        adapter_path = cfg["model"].get("adapter_path")
        if adapter_path and not Path(adapter_path).exists():
            logger.warning("Adapter path %s not found, using base model.", adapter_path)
            adapter_path = None
        hf_bundle = load_hf_model(
            model_name,
            cfg["model"].get("model_type", "seq2seq"),
            adapter_path,
        )

    for row in data_rows:
        if row["id"] in existing:
            continue
        question = row["question"]
        reference = normalize_reference(row.get("answer", ""))
        with timed() as tinfo:
            if provider in {"openai", "openrouter"}:
                messages = build_messages(prompt, question)
                response = chat_completion(
                    client=client,
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra=cfg["model"].get("extra"),
                )
                prediction = response["content"]
                usage = response.get("usage", {})
            elif provider == "ollama":
                base_url = cfg["model"].get("base_url", "http://localhost:11434")
                response = generate_ollama(model_name, build_messages(prompt, question), temperature, max_tokens, base_url)
                prediction = response["content"]
                usage = response.get("usage", {})
            elif provider == "hf":
                tokenizer, model = hf_bundle
                prompt_text = build_prompt_text(prompt, question)
                prediction = generate_hf(tokenizer, model, prompt_text, max_tokens, temperature)
                usage = {}
            else:
                raise ValueError(f"Unknown provider: {provider}")

        if postprocess:
            prediction = postprocess_one_sentence(prediction)

        meta = {
            "model": model_name,
            "provider": provider,
            "prompt_id": cfg["prompt"]["prompt_id"],
            "latency_ms": tinfo["elapsed_ms"],
            "token_usage": usage,
            "cost_estimate": estimate_cost(usage, pricing),
            "retrieval_info": {},
        }

        results.append(
            {
                "id": row["id"],
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "meta": meta,
            }
        )

    write_jsonl(results, output_path)
    logger.info("Saved %s predictions to %s", len(results), output_path)


if __name__ == "__main__":
    main()

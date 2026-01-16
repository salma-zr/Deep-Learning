from __future__ import annotations

import os
from typing import Any, Dict, Optional

from openai import OpenAI


def get_openai_client(base_url: Optional[str] = None, api_key_env: str = "OPENAI_API_KEY") -> OpenAI:
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in env var {api_key_env}.")
    return OpenAI(api_key=api_key, base_url=base_url)


def chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra:
        payload.update(extra)
    response = client.chat.completions.create(**payload)
    choice = response.choices[0].message.content or ""
    usage = response.usage.model_dump() if response.usage else {}
    return {"content": choice, "usage": usage}


def estimate_cost(
    usage: Dict[str, Any],
    pricing: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    if not pricing or not usage:
        return None
    prompt_cost = pricing.get("prompt_per_1k", 0.0)
    completion_cost = pricing.get("completion_per_1k", 0.0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    return (prompt_tokens / 1000.0) * prompt_cost + (completion_tokens / 1000.0) * completion_cost

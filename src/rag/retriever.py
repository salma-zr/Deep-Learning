import re
from typing import Dict, List

import requests


def chunk_text(text: str, chunk_size: int) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def retrieve_wikipedia(query: str, top_k: int = 3, lang: str = "en") -> List[Dict[str, str]]:
    search_url = f"https://{lang}.wikipedia.org/w/api.php"
    search_params = {"action": "query", "list": "search", "srsearch": query, "format": "json"}
    search_resp = requests.get(search_url, params=search_params, timeout=30)
    search_resp.raise_for_status()
    results = search_resp.json().get("query", {}).get("search", [])[:top_k]

    docs: List[Dict[str, str]] = []
    for result in results:
        title = result["title"]
        page_params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": True,
            "format": "json",
            "titles": title,
        }
        page_resp = requests.get(search_url, params=page_params, timeout=30)
        page_resp.raise_for_status()
        pages = page_resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            text = page.get("extract", "")
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            docs.append(
                {
                    "title": title,
                    "url": f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    "text": text,
                }
            )
    return docs


def retrieve_duckduckgo(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError("duckduckgo_search not installed. Install via pip if needed.") from exc

    docs: List[Dict[str, str]] = []
    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=top_k):
            text = result.get("body", "")
            if not text:
                continue
            docs.append({"title": result.get("title", ""), "url": result.get("href", ""), "text": text})
    return docs


def build_context(docs: List[Dict[str, str]], chunk_size: int, top_k: int) -> Dict[str, str]:
    chunks: List[str] = []
    sources: List[Dict[str, str]] = []
    for doc in docs:
        doc_chunks = chunk_text(doc["text"], chunk_size)
        for chunk in doc_chunks:
            chunks.append(chunk)
            sources.append({"title": doc.get("title", ""), "url": doc.get("url", "")})

    selected = chunks[:top_k]
    context = "\n\n".join(selected)
    return {"context": context, "sources": sources[: len(selected)]}

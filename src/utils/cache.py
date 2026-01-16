from pathlib import Path


def should_skip(output_path: str | Path, use_cache: bool) -> bool:
    if not use_cache:
        return False
    return Path(output_path).exists()

import re


SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    return SENTENCE_SPLIT.split(text)


def postprocess_one_sentence(text: str) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""
    sentence = sentences[0].strip()
    if sentence and sentence[-1] not in ".!?":
        sentence = sentence + "."
    return sentence


def count_sentences(text: str) -> int:
    return len(split_sentences(text))


def normalize_reference(text: str) -> str:
    return normalize_whitespace(text)

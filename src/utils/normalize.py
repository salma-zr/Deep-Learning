"""Text normalization utilities for answer processing."""

import re
from typing import Optional


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    # Replace multiple spaces/tabs with single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Replace multiple newlines with single newline
    text = re.sub(r'\n+', '\n', text)
    # Strip leading/trailing whitespace
    return text.strip()


def remove_markdown(text: str) -> str:
    """Remove common markdown formatting."""
    # Remove bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    
    # Remove code blocks
    text = re.sub(r'```[^`]*```', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # Remove bullet points
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    return text.strip()


def extract_first_sentence(text: str) -> str:
    """Extract the first sentence from text."""
    if not text:
        return ""
    
    text = text.strip()
    
    # Common sentence-ending patterns
    # Handle abbreviations like "e.g.", "i.e.", "Dr.", "vs." etc.
    abbreviations = r'(?<!\b(?:Dr|Mr|Mrs|Ms|Prof|vs|etc|e\.g|i\.e|ca|approx))'
    
    # Match sentence ending
    pattern = abbreviations + r'[.!?](?:\s|$)'
    
    match = re.search(pattern, text)
    if match:
        return text[:match.end()].strip()
    
    # If no sentence ending found, return the whole text
    return text


def is_single_sentence(text: str) -> bool:
    """Check if text appears to be a single sentence."""
    if not text:
        return True
    
    text = text.strip()
    
    # Count sentence-ending punctuation (excluding abbreviations)
    # This is a heuristic, not perfect
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    return len(sentences) <= 1


def count_sentences(text: str) -> int:
    """Count the approximate number of sentences in text."""
    if not text:
        return 0
    
    text = text.strip()
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Filter out empty strings
    sentences = [s for s in sentences if s.strip()]
    
    return max(1, len(sentences)) if text else 0


def normalize_answer(
    text: str,
    lowercase: bool = False,
    remove_punct: bool = False,
    single_sentence: bool = True,
    remove_md: bool = True,
) -> str:
    """
    Normalize an answer for evaluation.
    
    Args:
        text: Raw answer text
        lowercase: Convert to lowercase
        remove_punct: Remove punctuation
        single_sentence: Extract only first sentence
        remove_md: Remove markdown formatting
        
    Returns:
        Normalized answer
    """
    if not text:
        return ""
    
    # Basic cleanup
    text = text.strip()
    
    # Remove markdown
    if remove_md:
        text = remove_markdown(text)
    
    # Extract first sentence if needed
    if single_sentence:
        text = extract_first_sentence(text)
    
    # Normalize whitespace
    text = normalize_whitespace(text)
    
    # Lowercase
    if lowercase:
        text = text.lower()
    
    # Remove punctuation (keep spaces)
    if remove_punct:
        text = re.sub(r'[^\w\s]', '', text)
    
    return text.strip()


def is_insufficient_info(text: str) -> bool:
    """Check if the answer indicates insufficient information."""
    if not text:
        return True
    
    text = text.lower().strip()
    
    indicators = [
        "insufficient information",
        "not enough information",
        "cannot determine",
        "cannot answer",
        "unable to answer",
        "don't know",
        "do not know",
        "no information",
        "not specified",
        "not mentioned",
        "unclear",
        "unknown",
        "i don't have",
        "i cannot",
    ]
    
    return any(indicator in text for indicator in indicators)


def get_answer_stats(answers: list[str]) -> dict:
    """
    Compute statistics about a list of answers.
    
    Returns dict with:
        - avg_length: Average character length
        - avg_words: Average word count
        - multi_sentence_pct: Percentage with multiple sentences
        - empty_pct: Percentage empty/whitespace-only
        - insufficient_pct: Percentage indicating insufficient info
    """
    if not answers:
        return {
            "avg_length": 0,
            "avg_words": 0,
            "multi_sentence_pct": 0,
            "empty_pct": 100,
            "insufficient_pct": 0,
        }
    
    lengths = []
    word_counts = []
    multi_sentence = 0
    empty = 0
    insufficient = 0
    
    for answer in answers:
        if not answer or not answer.strip():
            empty += 1
            continue
        
        lengths.append(len(answer))
        word_counts.append(len(answer.split()))
        
        if not is_single_sentence(answer):
            multi_sentence += 1
        
        if is_insufficient_info(answer):
            insufficient += 1
    
    n = len(answers)
    return {
        "avg_length": sum(lengths) / max(len(lengths), 1),
        "avg_words": sum(word_counts) / max(len(word_counts), 1),
        "multi_sentence_pct": (multi_sentence / n) * 100,
        "empty_pct": (empty / n) * 100,
        "insufficient_pct": (insufficient / n) * 100,
    }

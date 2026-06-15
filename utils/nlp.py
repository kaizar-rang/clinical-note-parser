"""
Shared NLP helpers (regex utilities, NER wrappers).
"""
import re


LATERALITY_PATTERN = re.compile(
    r'\b(left|right|bilateral|midline)\b',
    re.IGNORECASE
)

SIZE_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*(mm|cm)',
    re.IGNORECASE
)


def extract_laterality(text: str) -> str | None:
    """Extract the first laterality mention from text."""
    match = LATERALITY_PATTERN.search(text)
    return match.group(1).lower() if match else None


def extract_size(text: str) -> dict | None:
    """Extract the first size measurement from text."""
    match = SIZE_PATTERN.search(text)
    if match:
        return {"value": float(match.group(1)), "unit": match.group(2).lower()}
    return None

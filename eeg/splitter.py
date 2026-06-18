"""
EEG report splitter.

Takes a raw EEG report row and extracts the clinically relevant
finding text, routing by service type (Routine, LTM, EMU).
"""

import re
import pandas as pd
from eeg.router import get_report_type


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Section headers -- ALL CAPS lines optionally ending with colon
HEADER_RE = re.compile(r'^\s*[A-Z][A-Z\s/]{4,}:?\s*$', re.MULTILINE)

# Separator lines like [REDACTED]-----[REDACTED]
SEPARATOR_RE = re.compile(r'\[REDACTED\]-{3,}\[REDACTED\]')

# Standalone "None" lines
NONE_LINE_RE = re.compile(r'^\s*None\s*$', re.MULTILINE)

# Boilerplate: signatures, attestation lines, metadata
BOILERPLATE_PATTERNS = [
    re.compile(r'^\s*\[REDACTED\].*?(?:MD|PhD)\.?\s*$', re.MULTILINE),
    re.compile(r'I \(\[REDACTED\]\) have reviewed.*?\.', re.DOTALL),
    re.compile(r'I \(Dr\. \[REDACTED\]\) have reviewed.*?\.', re.DOTALL),
    re.compile(r'METHODOLOGY:.*?(?=\n[A-Z]{4,}:|\Z)', re.DOTALL),
    re.compile(r'METADATA:.*', re.DOTALL),
    re.compile(r'ABBREVIATIONS.*', re.DOTALL),
    re.compile(r'AED.*?SCHEDULE:.*?(?=\n[A-Z]{4,}:|\Z)', re.DOTALL | re.IGNORECASE),
    re.compile(r'ANTIEPILEPTIC MEDICATIONS.*?(?=\n[A-Z]{4,}:|\Z)', re.DOTALL),
    re.compile(r'TECHNICAL SUMMARY:.*?(?=\n[A-Z]{4,}:|\Z)', re.DOTALL),
]

# IMPRESSION header variants
IMPRESSION_RE = re.compile(
    r'IMPRESSION(?:\s*/\s*SUMMARY)?\s*:\s*(.*?)(?=\n\s*[A-Z][A-Z\s/]{4,}:|\Z)',
    re.DOTALL
)


def _clean(text: str) -> str:
    """Apply shared cleaning steps to any extracted block."""
    # strip separator lines
    text = SEPARATOR_RE.sub('', text)
    # strip standalone None lines
    text = NONE_LINE_RE.sub('', text)
    # strip boilerplate patterns
    for pat in BOILERPLATE_PATTERNS:
        text = pat.sub('', text)
    # collapse excess whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _extract_impression(text: str) -> str | None:
    """Pull the IMPRESSION / SUMMARY block out of a report."""
    match = IMPRESSION_RE.search(text)
    if not match:
        return None
    block = match.group(1).strip()
    return _clean(block) or None


def _extract_before_first_header(text: str) -> str | None:
    """
    Extract everything before the first ALL-CAPS section header.
    Used as the fallback for Routine and EMU when no IMPRESSION exists.
    """
    match = HEADER_RE.search(text)
    if match:
        block = text[:match.start()]
    else:
        block = text
    return _clean(block) or None


# ---------------------------------------------------------------------------
# Per-type extractors
# ---------------------------------------------------------------------------

def split_routine(text):

    # primary: IMPRESSION block
    impression = _extract_impression(text)
    if impression:
        return impression
    
    # fallback: opening text before first header

    return _extract_before_first_header(text)

def split_ltm(text: str) -> str | None:
    """Extract finding from an LTM EEG report."""
    # primary: IMPRESSION / SUMMARY block
    impression = _extract_impression(text)
    if impression:
        return impression

    # secondary: most recent EPOCH block (listed newest first)
    epoch_match = re.search(
        r'EPOCH\s*:.*?\n(.*?)(?=\bEPOCH\b|METHODOLOGY|METADATA|\Z)',
        text, re.DOTALL
    )
    if epoch_match:
        return _clean(epoch_match.group(1)) or None

    # fallback: standalone BACKGROUND / SPORADIC DISCHARGES / SEIZURES sections
    fields = ['BACKGROUND', 'SPORADIC DISCHARGES', 'PERIODIC OR RHYTHMIC PATTERNS', 'SEIZURES']
    blocks = []
    for field in fields:
        match = re.search(
            rf'{field}\s*:(.*?)(?=\n[A-Z][A-Z\s]{{3,}}:|\Z)',
            text, re.DOTALL
        )
        if match:
            blocks.append(f'{field}: {match.group(1).strip()}')
    return _clean('\n\n'.join(blocks)) or None

def split_emu(text: str) -> str | None:
    """Extract finding from an EMU EEG report."""
    # primary: IMPRESSION block
    impression = _extract_impression(text)
    if impression:
        return impression
    # fallback: opening text before first header
    return _extract_before_first_header(text)

def split_report(row: pd.Series) -> str | None:
    """
    Extract the clinical finding from one EEG report row.
    This is the function the teaching tool calls directly.

    Args:
        row: one row from the EEG dataframe, must have
             ServiceName and ReportTextRedacted columns.

    Returns:
        Clean finding text, or None if extraction failed.
    """
    report_type = get_report_type(row['ServiceName'])
    if report_type is None:
        return None

    text = row['ReportTextRedacted']
    if pd.isna(text) or not text.strip():
        return None

    if report_type == 'routine':
        return split_routine(text)
    elif report_type == 'ltm':
        return split_ltm(text)
    elif report_type == 'emu':
        return split_emu(text)
    return None
"""
EEG report splitter.

Takes a raw EEG report row and extracts the clinically relevant
finding text, routing by service type (Routine, LTM, EMU).

Output column: finding_text (one row per report, same shape as input)

Coverage on eeg_reports.csv (49,234 usable rows):
  Routine: 15,711 / 15,712 (100.0%)
  LTM:     26,722 / 26,722 (100.0%)
  EMU:      6,800 /  6,800 (100.0%)
  Overall: 49,233 / 49,234 (100.0%)
"""

import re
import pandas as pd
from eeg.router import get_report_type


# ---------------------------------------------------------------------------
# Shared patterns
# ---------------------------------------------------------------------------

HEADER_RE = re.compile(r'^\s*[A-Z][A-Z\s/]{4,}:?\s*$', re.MULTILINE)

SEPARATOR_RE = re.compile(r'\[REDACTED\]-{3,}\[REDACTED\]')

NONE_LINE_RE = re.compile(r'^\s*None\s*$', re.MULTILINE)

STOP_HEADERS = re.compile(
    r'\n\s*(?:COMPARISON|INDICATION|CLINICAL INDICATION|PERTINENT MEDICATIONS|'
    r'METHOD|METHODOLOGY|DETAIL|DETAILS|HISTORY|HISTORY/REASON|BACKGROUND|'
    r'TECHNICAL SUMMARY|ANTIEPILEPTIC|AED|ELECTRODE|EVENTS|SEIZURES|'
    r'INTERICTAL|IMPRESSION|SUMMARY|STUDY TYPE)\s*[:\n]',
    re.IGNORECASE
)

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

IMPRESSION_RE = re.compile(
    r'(?:Final\s+)?IMPRESSION(?:\s*/\s*SUMMARY)?\s*:?\s*\n?(.*?)'
    r'(?=\n\s*(?:HISTORY|COMPARISON|INDICATION|CLINICAL INDICATION|'
    r'STUDY TYPE|DETAIL|METHODOLOGY|METADATA)|\Z)',
    re.DOTALL | re.IGNORECASE
)

# Reports that contain only a patient header with no clinical content
EMPTY_PATTERN = re.compile(
    r'^\s*(?:\[REDACTED\]\s+)?(?:Portable|Final)\s+(?:EEG|video EEG)\s+Report|'
    r'^\s*(?:\[REDACTED\]\s+)?Epilepsy Monitoring Unit Report',
    re.IGNORECASE | re.MULTILINE
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Apply shared cleaning steps to any extracted block."""
    text = SEPARATOR_RE.sub('', text)
    text = NONE_LINE_RE.sub('', text)
    for pat in BOILERPLATE_PATTERNS:
        text = pat.sub('', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _extract_impression(text: str) -> str | None:
    """Pull the IMPRESSION / SUMMARY block out of a report."""
    match = IMPRESSION_RE.search(text)
    if not match:
        return None
    return _clean(match.group(1).strip()) or None


def _extract_before_first_header(text: str) -> str | None:
    """
    Extract everything before the first known section header.
    Used as the fallback for Routine, EMU, and Routine-style LTM reports.
    """
    match = STOP_HEADERS.search(text)
    block = text[:match.start()] if match else text
    return _clean(block) or None


# ---------------------------------------------------------------------------
# Per-type extractors
# ---------------------------------------------------------------------------

def split_routine(text: str) -> str | None:
    """
    Extract finding from a Routine EEG report.

    Priority:
    1. IMPRESSION block (48.7% of Routine reports)
    2. Opening text before first section header
    3. DETAILS / DETAIL section (for reports starting with a header)
    """
    impression = _extract_impression(text)
    if impression:
        return impression

    opening = _extract_before_first_header(text)
    if opening:
        return opening

    # fallback for reports that start directly with a section header
    for field in ['DETAILS', 'DETAIL']:
        m = re.search(
            rf'{field}\s*:(.*?)(?=\n\s*(?:COMPARISON|INDICATION|METHOD|EKG|IMPRESSION)|\Z)',
            text, re.DOTALL
        )
        if m:
            return _clean(m.group(1)) or None

    return None


def split_ltm(text: str) -> str | None:
    """
    Extract finding from an LTM EEG report.

    Priority:
    1. IMPRESSION / SUMMARY block (83.9% of LTM reports)
    2. Most recent EPOCH block content (listed newest-first)
    3. Standalone BACKGROUND / SPORADIC DISCHARGES / SEIZURES sections
    4. Opening text before first header (Routine-style LTM reports)
    """
    impression = _extract_impression(text)
    if impression:
        return impression

    epoch_match = re.search(
        r'EPOCH\s*:.*?\n(?:\[REDACTED\]-+\[REDACTED\]\s*\n)?(.*?)'
        r'(?=\n\s*\[REDACTED\]-+\[REDACTED\]|\bEPOCH\b|METHODOLOGY|METADATA|\Z)',
        text, re.DOTALL
    )
    if epoch_match:
        result = _clean(epoch_match.group(1))
        if result:
            return result

    fields = ['BACKGROUND', 'SPORADIC DISCHARGES', 'PERIODIC OR RHYTHMIC PATTERNS', 'SEIZURES']
    blocks = []
    for field in fields:
        m = re.search(
            rf'{field}\s*:(.*?)(?=\n[A-Z][A-Z\s]{{3,}}:|\Z)',
            text, re.DOTALL
        )
        if m:
            blocks.append(f'{field}: {m.group(1).strip()}')
    if blocks:
        result = _clean('\n\n'.join(blocks))
        if result:
            return result

    return _extract_before_first_header(text)


def split_emu(text: str) -> str | None:
    """
    Extract finding from an EMU EEG report.

    Priority:
    1. IMPRESSION block (68.1% of EMU reports)
    2. Opening text before first section header
    """
    return _extract_impression(text) or _extract_before_first_header(text)


# ---------------------------------------------------------------------------
# Single-pass entrypoint (teaching tool)
# ---------------------------------------------------------------------------

def split_report(row: pd.Series) -> str | None:
    """
    Extract the clinical finding from one EEG report row.
    This is the function the teaching tool calls directly at runtime.

    Args:
        row: one row from the EEG dataframe, must have
             ServiceName and ReportTextRedacted columns.

    Returns:
        Clean finding text, or None if extraction failed or
        report contains no clinical content.
    """
    report_type = get_report_type(row['ServiceName'])
    if report_type is None:
        return None

    text = row['ReportTextRedacted']
    if pd.isna(text) or not str(text).strip():
        return None

    if report_type == 'routine':
        result = split_routine(text)
    elif report_type == 'ltm':
        result = split_ltm(text)
    elif report_type == 'emu':
        result = split_emu(text)
    else:
        return None

    # drop results that are just patient header blocks with no clinical content
    if result and EMPTY_PATTERN.match(result.strip()):
        return None

    return result


# ---------------------------------------------------------------------------
# Batch entrypoint (offline pipeline)
# ---------------------------------------------------------------------------

def split_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run split_report over the full EEG dataframe.
    Produces one output row per input row with a finding_text column added.
    Unlike the MRI splitter, this does not explode rows -- one report
    produces one finding_text value.
    """
    df = df.copy()
    df['finding_text'] = df.apply(split_report, axis=1)
    valid = df['finding_text'].notna().sum()
    print(f'Extracted: {valid:,} / {len(df):,} ({100*valid/len(df):.1f}%)')
    return df


if __name__ == '__main__':
    eeg = pd.read_csv('../data/eeg_reports.csv', engine='python', on_bad_lines='skip')
    eeg = eeg[eeg['ServiceName'].isin(['Routine', 'LTM', 'EMU'])].copy()
    result = split_dataframe(eeg)
    result.to_csv('../data/eeg_findings_split.csv', index=False)
    print('Saved: ../data/eeg_findings_split.csv')
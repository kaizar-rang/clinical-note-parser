"""
MRI report splitter.

Takes the raw `findings` column from mri_reports.csv and splits each report
into individual finding paragraphs, then filters noise and tags negatives
and extracranial paragraphs.

Pipeline:
    mri_reports.csv
        → load_structural()       -- filter to usable structural brain MRI rows
        → split_and_filter()      -- paragraph split + noise removal + tagging
        → mri_findings_split.csv  -- one row per finding, ready for extractor.py

Splitting strategy (validated via EDA on 36,924 reports):

    1. Multi-section detection: ~5.1% of reports combine BRAIN MRI + HEAD MRA
       + NECK MRA in one findings block. These are split on section headers and
       only the BRAIN MRI section is kept. Without this step, MRA vessel findings
       flood the output with non-brain content.

    2. Paragraph splitting: double newline (\n\n) is the primary boundary.
       99.2% of reports use this convention. Single-newline-only reports (0.8%)
       are left as one unit -- they are typically compact single-finding prose
       that the LLM extractor handles correctly without splitting.

    3. Minimum length filter: paragraphs under 30 chars are dropped. These are
       section sub-headers, lone punctuation, or stubs.

Noise filter categories (hard drop):
    - Too short (<30 chars)
    - Section headers: "BRAIN MRI:", "NECK MRA:", "ANTERIOR CIRCULATION"
    - Boilerplate/admin: automated system notifications, attending sign-off
    - Technical notes: "study terminated", "motion artifact limits evaluation"

Soft tags (kept but flagged for downstream filtering):
    - is_negative:     normal/negative findings, nothing to highlight in VXP
                       but useful as ground truth for the teaching tool
    - is_extracranial: non-brain structures (sinuses, mastoids, cervical vessels)
                       VXP cannot annotate these on a structural brain MRI
"""
import re
import pandas as pd
from utils.filters import is_noise, is_negative, is_extracranial

STRUCTURAL_TYPES = [
    'MRI BRAIN',
    'MRI BRAIN WITH AND WITHOUT CONTRAST',
    'MRI BRAIN WITHOUT CONTRAST',
]

# Matches the start of a new named section in multi-modality reports
SECTION_SPLIT_PATTERN = re.compile(
    r'\n(?=(?:BRAIN MRI|HEAD MRA|NECK MRA|MRI BRAIN AND NECK'
    r'|SPINE|CERVICAL|THORACIC|LUMBAR|STRUCTURAL MRI|FUNCTIONAL MRI)\s*[:\n])',
    re.IGNORECASE
)

# Matches brain-relevant section headers to keep
BRAIN_SECTION_PATTERN = re.compile(
    r'^(?:BRAIN MRI|MRI BRAIN|STRUCTURAL MRI)',
    re.IGNORECASE
)

# Inline section header at the start of a paragraph to strip
INLINE_HEADER_PATTERN = re.compile(
    r'^(?:BRAIN MRI|HEAD MRA|MRI BRAIN)\s*:?\s*\n',
    re.IGNORECASE
)


def load_structural(csv_path: str) -> pd.DataFrame:
    """
    Load mri_reports.csv and filter to usable structural brain MRI rows.

    Drops:
        - Rows with missing or stub findings ("not available", "none", "")
        - Non-structural scan types (PET, functional, angio, intraop)
    """
    mri = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')

    valid = mri[
        mri['findings'].notna() &
        ~mri['findings'].str.strip().str.lower().isin(['not available', 'none', ''])
    ]

    structural = valid[valid['Type'].isin(STRUCTURAL_TYPES)].copy()

    print(f'Total MRI rows:           {len(mri):,}')
    print(f'Usable rows:              {len(valid):,}')
    print(f'Structural brain MRI:     {len(structural):,}')
    print(f'Unique patients:          {structural["bdsp_patient_id"].nunique():,}')
    print(f'Unique sessions:          {structural["session_id"].nunique():,}')

    return structural


def _extract_brain_section(text: str) -> str:
    """
    For multi-section reports (BRAIN MRI + HEAD MRA + NECK MRA etc.),
    extract only the BRAIN MRI section. Returns the full text unchanged
    for single-section reports.
    """
    if not SECTION_SPLIT_PATTERN.search(text):
        return text

    sections = SECTION_SPLIT_PATTERN.split(text)
    brain_sections = [s for s in sections if BRAIN_SECTION_PATTERN.match(s.strip())]

    # If we found a brain section, use it. Otherwise fall back to full text
    # (avoids losing reports where the brain content comes before any header)
    return brain_sections[0] if brain_sections else sections[0]


def split_report(text: str) -> list[str]:
    """
    Split a single findings text into individual finding paragraphs.

    Strategy:
        1. Extract brain section only (handles multi-modality reports)
        2. Split on double newlines
        3. Strip inline section headers from paragraph starts
        4. Drop paragraphs under 30 chars
    """
    if pd.isna(text):
        return []

    text = _extract_brain_section(text)

    paragraphs = re.split(r'\n[ \t]*\n', text)

    result = []
    for p in paragraphs:
        p = p.strip()
        p = INLINE_HEADER_PATTERN.sub('', p).strip()
        if len(p) >= 30:
            result.append(p)

    return result


def split_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode each MRI report into individual finding paragraphs,
    apply noise filters, and tag negatives and extracranial findings.

    Args:
        df: Output of load_structural() -- one row per MRI session

    Returns:
        DataFrame with one row per finding paragraph, columns:
            bdsp_patient_id  -- patient identifier
            session_id       -- session identifier (one visit)
            Type             -- MRI scan type
            paragraph        -- individual finding text
            is_negative      -- True if normal/negative finding
            is_extracranial  -- True if describes non-brain structures
    """
    df = df.copy()

    df['paragraph_list'] = df['findings'].apply(split_report)
    exploded = (
        df.explode('paragraph_list')
        .rename(columns={'paragraph_list': 'paragraph'})
        .dropna(subset=['paragraph'])
        .reset_index(drop=True)
    )

    # Hard drop: noise paragraphs
    noise_mask = exploded['paragraph'].apply(is_noise)
    filtered = exploded[~noise_mask].copy().reset_index(drop=True)

    # Soft tags
    filtered['is_negative'] = filtered['paragraph'].apply(is_negative)
    filtered['is_extracranial'] = filtered['paragraph'].apply(is_extracranial)

    positive = (~filtered['is_negative'] & ~filtered['is_extracranial']).sum()

    print(f'\nSplit summary:')
    print(f'  Total paragraphs:            {len(exploded):,}')
    print(f'  Dropped (noise):             {noise_mask.sum():,}')
    print(f'  Remaining:                   {len(filtered):,}')
    print(f'    - Negative/normal:         {filtered["is_negative"].sum():,}')
    print(f'    - Extracranial:            {filtered["is_extracranial"].sum():,}')
    print(f'    - Positive brain findings: {positive:,}')

    return filtered


if __name__ == "__main__":
    import os
    from config import MRI_CSV

    os.makedirs('data', exist_ok=True)

    structural = load_structural(MRI_CSV)
    findings = split_and_filter(structural)

    out_cols = ['bdsp_patient_id', 'session_id', 'Type', 'paragraph', 'is_negative', 'is_extracranial']
    out_path = 'data/mri_findings_split.csv'
    findings[out_cols].to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')
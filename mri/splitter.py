"""
MRI report splitter.

Takes the raw `findings` column from mri_reports.csv and splits each report
into individual findings (one paragraph = one finding), then filters noise
and tags negatives and extracranial paragraphs.
"""
import re
import pandas as pd
from utils.filters import is_noise, is_negative, is_extracranial

STRUCTURAL_TYPES = [
    'MRI BRAIN',
    'MRI BRAIN WITH AND WITHOUT CONTRAST',
    'MRI BRAIN WITHOUT CONTRAST',
]


def load_structural(csv_path: str) -> pd.DataFrame:
    """Load and filter to usable structural MRI rows."""
    mri = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
    valid = mri[
        mri['findings'].notna() &
        ~mri['findings'].str.strip().str.lower().isin(['not available', 'none', ''])
    ]
    structural = valid[valid['Type'].isin(STRUCTURAL_TYPES)].copy()
    print(f'Structural MRI rows loaded: {len(structural):,}')
    return structural


def split_report(text: str) -> list[str]:
    """Split a findings text into individual paragraphs."""
    if pd.isna(text):
        return []
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]


def split_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode each report into individual findings and apply noise filters.

    Returns a DataFrame with one finding per row, tagged with:
        - is_negative:     True if finding is a normal/negative statement
        - is_extracranial: True if finding describes non-brain structures
    """
    df = df.copy()
    df['paragraph_list'] = df['findings'].apply(split_report)
    exploded = (
        df.explode('paragraph_list')
        .rename(columns={'paragraph_list': 'paragraph'})
        .dropna(subset=['paragraph'])
        .reset_index(drop=True)
    )

    # Hard drop: noise
    noise_mask = exploded['paragraph'].apply(is_noise)
    filtered = exploded[~noise_mask].copy().reset_index(drop=True)

    # Soft tags
    filtered['is_negative'] = filtered['paragraph'].apply(is_negative)
    filtered['is_extracranial'] = filtered['paragraph'].apply(is_extracranial)

    print(f'Total paragraphs after split: {len(exploded):,}')
    print(f'Dropped (noise):              {noise_mask.sum():,}')
    print(f'Remaining:                    {len(filtered):,}')
    print(f'  - Negative/normal:          {filtered["is_negative"].sum():,}')
    print(f'  - Extracranial:             {filtered["is_extracranial"].sum():,}')
    print(f'  - Positive brain findings:  {(~filtered["is_negative"] & ~filtered["is_extracranial"]).sum():,}')

    return filtered


if __name__ == "__main__":
    from config import MRI_CSV
    structural = load_structural(MRI_CSV)
    findings = split_and_filter(structural)
    out_cols = ['bdsp_patient_id', 'session_id', 'Type', 'paragraph', 'is_negative', 'is_extracranial']
    findings[out_cols].to_csv('data/mri_findings_split.csv', index=False)
    print('Saved: data/mri_findings_split.csv')

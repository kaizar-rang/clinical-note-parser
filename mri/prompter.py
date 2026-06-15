"""
MRI VXP prompt generator.

Takes extracted MRI findings (output of extractor.py) and formats them
into natural language prompts for VoxelPrompt (VXP).
"""
import json
import pandas as pd


def build_vxp_prompt(extraction: dict) -> str | None:
    """
    Convert a structured extraction dict into a VXP natural language prompt.
    Returns None if insufficient info to build a prompt.
    """
    region = extraction.get("region")
    finding_type = extraction.get("finding_type")

    if not region and not finding_type:
        return None

    parts = []

    if finding_type:
        parts.append(finding_type)
    else:
        parts.append("abnormality")

    if region:
        parts.append(f"in the {region}")

    size = extraction.get("size")
    if size:
        parts.append(f"measuring {size['value']} {size['unit']}")

    temporal = extraction.get("temporal")
    if temporal and temporal != "null":
        parts.append(f"({temporal})")

    prompt = "Highlight " + " ".join(parts) + "."
    return prompt


def add_prompts(df: pd.DataFrame) -> pd.DataFrame:
    """Add a vxp_prompt column to the extracted findings DataFrame."""
    df = df.copy()

    def safe_build(row):
        if pd.isna(row.get('extraction')):
            return None
        try:
            extraction = json.loads(row['extraction'])
            return build_vxp_prompt(extraction)
        except Exception:
            return None

    df['vxp_prompt'] = df.apply(safe_build, axis=1)
    valid = df['vxp_prompt'].notna().sum()
    print(f'VXP prompts generated: {valid:,} / {len(df):,}')
    return df


if __name__ == "__main__":
    extracted = pd.read_csv('data/mri_findings_extracted.csv')
    prompted = add_prompts(extracted)
    prompted.to_csv('data/mri_findings_prompted.csv', index=False)
    print('Saved: data/mri_findings_prompted.csv')

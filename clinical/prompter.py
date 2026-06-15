"""
Clinical note VXP prompt generator.

Formats clinical extractions into VXP prompts based on suspected region
derived from symptom-to-anatomy mapping.
"""
import json
import pandas as pd


def build_vxp_prompt(extraction: dict) -> str | None:
    """Convert clinical extraction to a VXP region prompt."""
    region = extraction.get("suspected_region")
    pathology = extraction.get("suspected_pathology")

    if not region:
        return None

    if pathology and pathology != "null":
        return f"Highlight suspected {pathology} in the {region}."
    return f"Highlight suspected abnormality in the {region}."


def add_prompts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def safe_build(row):
        if pd.isna(row.get('extraction')):
            return None
        try:
            return build_vxp_prompt(json.loads(row['extraction']))
        except Exception:
            return None

    df['vxp_prompt'] = df.apply(safe_build, axis=1)
    print(f'VXP prompts generated: {df["vxp_prompt"].notna().sum():,} / {len(df):,}')
    return df

"""
EEG VXP prompt generator.

Formats EEG extractions into natural language prompts for VoxelPrompt.
"""
import json
import pandas as pd


def build_vxp_prompt(extraction: dict) -> str | None:
    """Convert EEG extraction to a VXP prompt."""
    parts = []

    focal = extraction.get("focal_findings")
    laterality = extraction.get("laterality")
    epileptiform = extraction.get("epileptiform_activity")
    slowing = extraction.get("slowing")

    if epileptiform and epileptiform != "null":
        loc = f" over the {laterality} hemisphere" if laterality and laterality != "null" else ""
        parts.append(f"Highlight region of {epileptiform}{loc}.")

    if slowing and slowing != "null":
        loc = f" in the {laterality} hemisphere" if laterality and laterality != "null" else ""
        parts.append(f"Highlight region of {slowing} slowing{loc}.")

    if focal:
        for f in focal:
            parts.append(f"Highlight focal abnormality: {f}.")

    return " ".join(parts) if parts else None


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

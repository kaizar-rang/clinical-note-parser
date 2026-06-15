"""
MRI finding feature extractor.

Takes individual finding paragraphs (output of splitter.py) and uses an LLM
to extract all clinically relevant information as structured JSON.
"""
import json
import pandas as pd
from tqdm import tqdm
from utils.llm import call_llm, parse_json_response

SYSTEM_PROMPT = """You are a clinical NLP assistant specializing in neuroradiology.
Extract all clinically relevant information from the given MRI finding.
Return valid JSON only. No markdown, no explanation."""

EXTRACTION_PROMPT = """Extract all clinically relevant information from this MRI finding.

Finding:
{finding}

Return a JSON object with any of the following fields that are present:
{{
    "region": "anatomical region (e.g. left occipital lobe, right basal ganglia)",
    "laterality": "left | right | bilateral | midline | null",
    "finding_type": "type of pathology (e.g. infarct, hemorrhage, edema, lesion, atrophy)",
    "size": {{"value": number, "unit": "mm | cm"}} or null,
    "signal_characteristics": ["list of MRI signal findings e.g. T2 hyperintense, FLAIR abnormality"],
    "enhancement": "enhancing | non-enhancing | null",
    "temporal": "new | unchanged | increased | decreased | resolved | null",
    "severity": "mild | moderate | severe | null",
    "additional": "any other clinically relevant detail not captured above"
}}

Omit fields that are not present in the finding. Return only the JSON object."""


def extract_finding(text: str) -> dict:
    """Run LLM extraction on a single finding paragraph."""
    try:
        prompt = EXTRACTION_PROMPT.format(finding=text)
        response = call_llm(prompt, system=SYSTEM_PROMPT, json_mode=True)
        return parse_json_response(response)
    except Exception as e:
        return {"error": str(e), "raw": text}


def extract_dataframe(df: pd.DataFrame, text_col: str = "paragraph") -> pd.DataFrame:
    """
    Run extraction on all positive brain findings in a DataFrame.
    Skips negatives and extracranials.
    """
    mask = ~df['is_negative'] & ~df['is_extracranial']
    df = df.copy()
    df['extraction'] = None

    results = []
    for idx, row in tqdm(df[mask].iterrows(), total=mask.sum(), desc="Extracting"):
        results.append((idx, extract_finding(row[text_col])))

    for idx, result in results:
        df.at[idx, 'extraction'] = json.dumps(result)

    return df


if __name__ == "__main__":
    findings = pd.read_csv('data/mri_findings_split.csv')
    extracted = extract_dataframe(findings)
    extracted.to_csv('data/mri_findings_extracted.csv', index=False)
    print('Saved: data/mri_findings_extracted.csv')

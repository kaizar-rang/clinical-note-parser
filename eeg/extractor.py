"""
EEG finding feature extractor.

Uses an LLM to extract clinically relevant information from EEG report text.
Handles Routine, LTM, and EMU report formats.
"""
import json
import pandas as pd
from tqdm import tqdm
from utils.llm import call_llm, parse_json_response

SYSTEM_PROMPT = """You are a clinical NLP assistant specializing in clinical neurophysiology.
Extract all clinically relevant information from the given EEG report.
Return valid JSON only. No markdown, no explanation."""

EXTRACTION_PROMPT = """Extract all clinically relevant information from this EEG report.

Report:
{report}

Return a JSON object with any of the following fields that are present:
{{
    "background_rhythm": "description of background EEG activity",
    "dominant_frequency": {{"value": number, "unit": "Hz"}} or null,
    "laterality": "left | right | bilateral | diffuse | null",
    "focal_findings": ["list of focal abnormalities with electrode locations"],
    "epileptiform_activity": "type of epileptiform discharge (e.g. spike-wave, sharp waves) or null",
    "seizures": true | false | null,
    "seizure_type": "seizure classification if present or null",
    "slowing": "focal | generalized | null",
    "impression": "overall clinical impression from the report",
    "additional": "any other clinically relevant detail"
}}

Omit fields that are not present. Return only the JSON object."""


def extract_report(text: str) -> dict:
    """Run LLM extraction on a single EEG report."""
    try:
        prompt = EXTRACTION_PROMPT.format(report=text[:4000])  # truncate for token limit
        response = call_llm(prompt, system=SYSTEM_PROMPT, json_mode=True)
        return parse_json_response(response)
    except Exception as e:
        return {"error": str(e), "raw": text[:200]}


def extract_dataframe(df: pd.DataFrame, text_col: str = 'ReportTextRedacted') -> pd.DataFrame:
    """Run extraction on all EEG reports in a DataFrame."""
    df = df.copy()
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting EEG"):
        results.append((idx, extract_report(row[text_col])))
    for idx, result in results:
        df.at[idx, 'extraction'] = json.dumps(result)
    return df


if __name__ == "__main__":
    import pandas as pd
    from eeg.router import route
    from config import EEG_CSV

    eeg = pd.read_csv(EEG_CSV, engine='python', on_bad_lines='skip')
    subsets = route(eeg)

    all_extracted = []
    for service, subset in subsets.items():
        print(f'\nExtracting {service}...')
        extracted = extract_dataframe(subset)
        extracted['service_type'] = service
        all_extracted.append(extracted)

    combined = pd.concat(all_extracted).reset_index(drop=True)
    combined.to_csv('data/eeg_extracted.csv', index=False)
    print('Saved: data/eeg_extracted.csv')

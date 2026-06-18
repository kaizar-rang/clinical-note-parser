"""
MRI finding feature extractor.

Takes individual finding paragraphs (output of splitter.py) and uses an LLM
to extract all clinically relevant information as structured JSON.
"""
import json
import pandas as pd
from tqdm import tqdm
from utils.llm import call_llm, parse_json_response
from concurrent.futures import ThreadPoolExecutor

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


def extract_dataframe(
    df: pd.DataFrame,
    text_col: str = "paragraph",
    output_path: str = "data/mri_findings_extracted.csv",
    checkpoint_every: int = 500,
    max_workers: int = 16,
) -> pd.DataFrame:
    mask = ~df['is_negative'] & ~df['is_extracranial']
    to_process = df[mask].copy()

    # Resume: skip already-processed rows
    processed_ids = set()
    try:
        existing = pd.read_csv(output_path)
        processed_ids = set(existing['index'].tolist())
        print(f"Resuming: {len(processed_ids)} already done")
    except FileNotFoundError:
        pass

    to_process = to_process[~to_process.index.isin(processed_ids)]

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_finding, row[text_col]): idx
            for idx, row in to_process.iterrows()
        }
        for i, future in enumerate(tqdm(futures, total=len(futures), desc="Extracting")):
            idx = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {"error": str(e)}
            results.append({**to_process.loc[idx].to_dict(), "index": idx, "extraction": json.dumps(result)})

            if (i + 1) % checkpoint_every == 0:
                _flush(results, output_path, write_header=not processed_ids and i < checkpoint_every)
                results = []

    if results:
        _flush(results, output_path, write_header=not processed_ids and len(results) == len(futures))

    return pd.read_csv(output_path)


def _flush(rows: list[dict], path: str, write_header: bool):
    pd.DataFrame(rows).to_csv(path, mode='a', header=write_header, index=False)


if __name__ == "__main__":
    findings = pd.read_csv('data/mri_findings_split.csv')
    extracted = extract_dataframe(findings)
    extracted.to_csv('data/mri_findings_extracted.csv', index=False)
    print('Saved: data/mri_findings_extracted.csv')
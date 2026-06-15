"""
Clinical note feature extractor.

Extracts symptoms, complaints, and clinical context from hospital admission notes.
These drive the localization hypothesis (symptom → suspected brain region).
"""
import json
import pandas as pd
from tqdm import tqdm
from utils.llm import call_llm, parse_json_response

SYSTEM_PROMPT = """You are a clinical NLP assistant. Extract structured information
from hospital admission notes and clinical complaints.
Return valid JSON only. No markdown, no explanation."""

EXTRACTION_PROMPT = """Extract all clinically relevant information from this clinical note.

Note:
{note}

Return a JSON object with any of the following fields that are present:
{{
    "chief_complaint": "primary reason for admission in plain language",
    "symptoms": ["list of individual symptoms reported"],
    "symptom_laterality": "left | right | bilateral | null",
    "onset": "acute | subacute | chronic | null",
    "duration": "duration of symptoms if stated",
    "suspected_region": "brain region the symptoms suggest (e.g. right occipital lobe, brainstem)",
    "suspected_pathology": "suspected pathology type if implied (e.g. stroke, tumor, seizure)",
    "relevant_history": "relevant past medical history",
    "additional": "any other clinically relevant detail"
}}

Omit fields that are not present. Return only the JSON object."""


def extract_note(text: str) -> dict:
    """Run LLM extraction on a single clinical note."""
    try:
        prompt = EXTRACTION_PROMPT.format(note=text[:4000])
        response = call_llm(prompt, system=SYSTEM_PROMPT, json_mode=True)
        return parse_json_response(response)
    except Exception as e:
        return {"error": str(e), "raw": text[:200]}


def extract_dataframe(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Run extraction on all clinical notes in a DataFrame."""
    df = df.copy()
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting clinical"):
        df.at[idx, 'extraction'] = json.dumps(extract_note(row[text_col]))
    return df

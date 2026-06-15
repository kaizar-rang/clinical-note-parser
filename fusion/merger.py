"""
Per-patient merger.

Combines MRI, EEG, and clinical extractions for the same patient
into a single structured record ready for VXP prompt assembly.
"""
import json
import pandas as pd


def merge_patient(
    patient_id: str,
    mri_rows: pd.DataFrame,
    eeg_rows: pd.DataFrame,
    clinical_rows: pd.DataFrame,
) -> dict:
    """
    Merge all extractions for one patient into a unified dict.

    Args:
        patient_id:    Patient identifier
        mri_rows:      Rows from mri_findings_prompted.csv for this patient
        eeg_rows:      Rows from eeg_extracted.csv for this patient
        clinical_rows: Rows from clinical notes for this patient

    Returns:
        Dict with keys: patient_id, mri_findings, eeg_findings, clinical_findings
    """
    def safe_parse(val):
        if pd.isna(val):
            return {}
        try:
            return json.loads(val)
        except Exception:
            return {}

    mri_findings = [
        {
            "session_id": row.get("session_id"),
            "extraction": safe_parse(row.get("extraction")),
            "vxp_prompt": row.get("vxp_prompt"),
        }
        for _, row in mri_rows.iterrows()
        if not row.get("is_negative") and not row.get("is_extracranial")
    ]

    eeg_findings = [
        {
            "session_id": row.get("SessionID"),
            "service_type": row.get("service_type"),
            "extraction": safe_parse(row.get("extraction")),
            "vxp_prompt": row.get("vxp_prompt"),
        }
        for _, row in eeg_rows.iterrows()
    ]

    clinical_findings = [
        {
            "extraction": safe_parse(row.get("extraction")),
            "vxp_prompt": row.get("vxp_prompt"),
        }
        for _, row in clinical_rows.iterrows()
    ]

    return {
        "patient_id": patient_id,
        "mri_findings": mri_findings,
        "eeg_findings": eeg_findings,
        "clinical_findings": clinical_findings,
    }


def merge_all(
    mri_df: pd.DataFrame,
    eeg_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    patient_id_col_mri: str = "bdsp_patient_id",
    patient_id_col_eeg: str = "BDSPPatientID",
    patient_id_col_clinical: str = "patient_id",
) -> list[dict]:
    """Merge all patients. Returns a list of per-patient merged dicts."""
    all_ids = set(mri_df[patient_id_col_mri].unique())
    merged = []
    for pid in all_ids:
        mri_rows = mri_df[mri_df[patient_id_col_mri] == pid]
        eeg_rows = eeg_df[eeg_df[patient_id_col_eeg] == pid] if len(eeg_df) else pd.DataFrame()
        clinical_rows = clinical_df[clinical_df[patient_id_col_clinical] == pid] if len(clinical_df) else pd.DataFrame()
        merged.append(merge_patient(pid, mri_rows, eeg_rows, clinical_rows))
    print(f'Merged {len(merged):,} patients')
    return merged

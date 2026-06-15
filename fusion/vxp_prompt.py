"""
Final VXP prompt assembler.

Takes a merged per-patient dict and builds a single unified prompt
combining MRI, EEG, and clinical findings for VoxelPrompt.
"""


def build_unified_prompt(patient_record: dict) -> str:
    """
    Assemble a unified VXP prompt from all three note types for one patient.

    Args:
        patient_record: Output of fusion/merger.py merge_patient()

    Returns:
        Natural language prompt string for VXP
    """
    sections = []

    # Clinical context first -- drives the localization hypothesis
    clinical = patient_record.get("clinical_findings", [])
    if clinical:
        clinical_prompts = [c["vxp_prompt"] for c in clinical if c.get("vxp_prompt")]
        if clinical_prompts:
            sections.append("Clinical: " + " ".join(clinical_prompts))

    # MRI findings -- ground truth structural pathology
    mri = patient_record.get("mri_findings", [])
    if mri:
        mri_prompts = [m["vxp_prompt"] for m in mri if m.get("vxp_prompt")]
        if mri_prompts:
            sections.append("MRI: " + " ".join(mri_prompts))

    # EEG findings -- functional correlate
    eeg = patient_record.get("eeg_findings", [])
    if eeg:
        eeg_prompts = [e["vxp_prompt"] for e in eeg if e.get("vxp_prompt")]
        if eeg_prompts:
            sections.append("EEG: " + " ".join(eeg_prompts))

    return "\n".join(sections) if sections else ""


def build_all_prompts(merged_records: list[dict]) -> list[dict]:
    """Build unified VXP prompts for all patients."""
    results = []
    for record in merged_records:
        prompt = build_unified_prompt(record)
        results.append({
            "patient_id": record["patient_id"],
            "vxp_prompt": prompt,
        })
    valid = sum(1 for r in results if r["vxp_prompt"])
    print(f'Unified prompts built: {valid:,} / {len(results):,}')
    return results

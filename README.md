# cdac-parser

Clinical note parsing pipeline for the CDAC medical education teaching tool.

Parses MRI reports, EEG reports, and clinical admission notes into structured
extractions, then merges them per patient into a unified VoxelPrompt (VXP) prompt
for brain region annotation and overlay generation.

## Pipeline

```
MRI report   ──► splitter ──► extractor ──► prompter ──┐
EEG report   ──►            extractor ──► prompter ──┤──► fusion/merger ──► VXP prompt
Clinical note──►            extractor ──► prompter ──┘
```

## Setup

```bash
pip install -r requirements.txt
```

## Structure

- `mri/`      — MRI report splitting, extraction, prompt formatting
- `eeg/`      — EEG report routing, extraction, prompt formatting
- `clinical/` — Clinical note extraction, prompt formatting
- `fusion/`   — Per-patient merging and final VXP prompt assembly
- `utils/`    — Shared LLM wrapper, NLP helpers, noise filters
- `notebooks/`— EDA notebooks

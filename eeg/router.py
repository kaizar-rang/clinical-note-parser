# eeg/router.py

import pandas as pd

SERVICE_MAP = {
    "Routine": "routine",
    "LTM":     "ltm",
    "EMU":     "emu",
}

def get_report_type(service_name: str) -> str | None:
    """
    Map a single ServiceName value to a report type string.
    Returns None for unsupported service types.
    Used by the single-pass teaching tool entrypoint.
    """
    return SERVICE_MAP.get(service_name, None)


def route(df: pd.DataFrame, service_col: str = 'ServiceName') -> dict[str, pd.DataFrame]:
    """
    Split EEG DataFrame into subsets by service type.
    Returns dict with keys 'routine', 'ltm', 'emu' mapping to DataFrames.
    Used by the batch pipeline.
    """
    subsets = {}
    for service, label in SERVICE_MAP.items():
        subset = df[df[service_col] == service].copy()
        subsets[label] = subset
        print(f'{service}: {len(subset):,} reports')

    return subsets

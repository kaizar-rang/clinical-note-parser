"""
EEG report router.

Classifies each EEG report into its service type (Routine, LTM, EMU)
since each has a different structure requiring different extraction logic.
"""
import pandas as pd

SERVICE_TYPES = ['Routine', 'LTM', 'EMU']


def route(df: pd.DataFrame, service_col: str = 'ServiceName') -> dict[str, pd.DataFrame]:
    """
    Split EEG DataFrame into subsets by service type.

    Returns:
        Dict with keys 'Routine', 'LTM', 'EMU' mapping to DataFrames.
    """
    subsets = {}
    for service in SERVICE_TYPES:
        subset = df[df[service_col].str.contains(service, case=False, na=False)].copy()
        subsets[service] = subset
        print(f'{service}: {len(subset):,} reports')
    return subsets

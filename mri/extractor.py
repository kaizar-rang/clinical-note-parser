import asyncio
from concurrent.futures import ThreadPoolExecutor
import csv

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
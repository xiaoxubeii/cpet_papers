from __future__ import annotations
from typing import Dict, Any
import pandas as pd

REQUIRED_CASE_COLS = ['case_id', 'center']
OPTIONAL_CASE_COLS = ['time_at_at_s']


def load_cases(cases_csv: str) -> pd.DataFrame:
    df = pd.read_csv(cases_csv)
    missing = [c for c in REQUIRED_CASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"cases_csv missing columns: {missing}")
    # normalize column names
    return df


def load_timeseries(path: str) -> pd.DataFrame:
    """Deprecated: CSV loader; kept for backward compatibility when data is not in HDF5."""
    df = pd.read_csv(path)
    if 'Time' not in df.columns:
        raise ValueError(f"timeseries {path} missing 'Time' column")
    return df


def append_row_csv(path: str, row: Dict[str, Any]) -> None:
    import os
    import csv
    exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)

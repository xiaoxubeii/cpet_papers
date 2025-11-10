"""
Risk-aware case assignment generator.
- Inputs: cases.csv, readers.csv
- Optional: compute risk score by deriving VO2/kg@AT (needs Time_at_AT + timeseries paths)
- Output: assignment.csv (each reader ~40 cases), balanced by center and risk
Run: python make_assignments.py --cases cases.csv --readers readers.csv --out assignment.csv --per-reader 40 --h5 /path/to/cpet_data_source_xuhui.h5
"""
from __future__ import annotations
import argparse
import math
import random
from typing import List
import numpy as np
import pandas as pd

try:
    from data_io import load_cases  # type: ignore
    from sop_at_value import derive_values_from_timeseries  # type: ignore
    from h5_backend import H5CaseSource  # type: ignore
except Exception:
    from .data_io import load_cases  # type: ignore
    from .sop_at_value import derive_values_from_timeseries  # type: ignore
    from .h5_backend import H5CaseSource  # type: ignore


def compute_risk(cases: pd.DataFrame, h5src: H5CaseSource) -> pd.DataFrame:
    rows = []
    for _, r in cases.iterrows():
        risk = 0
        try:
            ts = h5src.get_timeseries(str(r['case_id']))
            time_at = r.get('time_at_at_s')
            if not (isinstance(time_at, (int, float))):
                time_at = h5src.get_time_at_at(str(r['case_id']))
            d = derive_values_from_timeseries(ts, time_at, window_s=10.0)
            vo2 = d['VO2_kg_at_AT']
            cov = d['coverage_ratio'] or 0.0
            warnings = d['warnings'] or ''
            if np.isfinite(vo2) and (abs(vo2 - 11.0) <= 0.3 or abs(vo2 - 14.0) <= 0.3):
                risk += 2
            if cov < 0.8 or warnings:
                risk += 1
        except Exception:
            risk += 1
        rows.append({'case_id': r['case_id'], 'risk': risk})
    score = pd.DataFrame(rows)
    return cases.merge(score, on='case_id', how='left')


def assign_cases(cases: pd.DataFrame, readers: pd.DataFrame, per_reader: int) -> pd.DataFrame:
    # ensure risk exists
    if 'risk' not in cases.columns:
        cases['risk'] = 0
    # stratify by center, then sort by risk desc and round-robin to readers
    groups = []
    for center, g in cases.groupby('center'):
        g = g.sort_values(['risk'], ascending=[False]).reset_index(drop=True)
        groups.append(g)
    pool = pd.concat(groups, axis=0).reset_index(drop=True)

    assignments = []
    reader_ids: List[str] = readers['reader_id'].astype(str).tolist()
    quotas = {rid: per_reader for rid in reader_ids}

    i = 0
    for _, row in pool.iterrows():
        available = [rid for rid, q in quotas.items() if q > 0]
        if not available:
            break
        rid = reader_ids[i % len(reader_ids)]
        # pick next available rid in cyclic order
        j = 0
        while quotas.get(rid, 0) == 0 and j < len(reader_ids):
            i += 1
            rid = reader_ids[i % len(reader_ids)]
            j += 1
        if quotas.get(rid, 0) == 0:
            continue
        batch_id = 1 + ((per_reader - quotas[rid]) // 20)  # 20 per batch
        assignments.append({
            'case_id': row['case_id'],
            'reader_id': rid,
            'center': row.get('center', ''),
            'risk': int(row.get('risk', 0)),
            'batch_id': int(batch_id),
        })
        quotas[rid] -= 1
        i += 1

    return pd.DataFrame(assignments)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', required=True)
    ap.add_argument('--readers', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--per-reader', type=int, default=40)
    ap.add_argument('--h5', required=True)
    ap.add_argument('--compute-risk', action='store_true', help='Derive VO2@AT to compute risk')
    args = ap.parse_args()

    cases = load_cases(args.cases)
    h5src = H5CaseSource(args.h5)
    readers = pd.read_csv(args.readers)
    if args.compute_risk:
        cases = compute_risk(cases, h5src)
    asg = assign_cases(cases, readers, args.per_reader)
    asg.to_csv(args.out, index=False)
    print(f"Wrote {len(asg)} assignments -> {args.out}")


if __name__ == '__main__':
    main()

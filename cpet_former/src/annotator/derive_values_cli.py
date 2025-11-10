"""
CLI to derive reads_human_with_values.csv from reads_human.csv and HDF5 timeseries.
Usage: python derive_values_cli.py --cases cases.csv --h5 cpet_data_source_xxx.h5 --reads reads_human.csv --out reads_human_with_values.csv
"""
from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
try:
    from data_io import load_cases  # type: ignore
    from sop_at_value import derive_values_from_timeseries  # type: ignore
    from h5_backend import H5CaseSource  # type: ignore
except Exception:
    from .data_io import load_cases  # type: ignore
    from .sop_at_value import derive_values_from_timeseries  # type: ignore
    from .h5_backend import H5CaseSource  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', required=True)
    ap.add_argument('--reads', required=True)
    ap.add_argument('--h5', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--window', type=float, default=10.0)
    args = ap.parse_args()

    cases = load_cases(args.cases)
    h5src = H5CaseSource(args.h5)
    reads = pd.read_csv(args.reads)
    df = reads.merge(cases[['case_id']], on='case_id', how='left')

    rows = []
    for _, r in df.iterrows():
        ts = h5src.get_timeseries(str(r['case_id']))
        d = derive_values_from_timeseries(ts, r['time_at_at_s'], window_s=args.window)
        row = dict(r)
        row.update({
            'vo2_kg_at_at': d['VO2_kg_at_AT'],
            'hr_at_at': d['HR_at_AT'],
            'workload_at_at': d['Workload_at_AT'],
            'coverage_ratio': d['coverage_ratio'],
            'warnings': d['warnings'],
        })
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows -> {args.out}")


if __name__ == '__main__':
    main()

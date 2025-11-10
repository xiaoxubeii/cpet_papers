"""
Generate cases.csv from a CPET HDF5 source (cpet_data_source_<center>.h5).
It extracts case_id (Examination_ID), center (arg), and Time_at_AT from summary.

Usage:
  python generate_cases_from_h5.py --h5 /path/to/cpet_data_source_xuhui.h5 --center xuhui --out cases.csv
"""
from __future__ import annotations
import argparse
import pandas as pd
try:
    from h5_backend import H5CaseSource  # type: ignore
except Exception:
    from .h5_backend import H5CaseSource  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--center', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    h5src = H5CaseSource(args.h5)
    rows = []
    for exam_id in h5src.list_exam_ids():
        t_at = h5src.get_time_at_at(exam_id)
        rows.append({'case_id': exam_id, 'center': args.center, 'time_at_at_s': t_at})
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows -> {args.out}")


if __name__ == '__main__':
    main()


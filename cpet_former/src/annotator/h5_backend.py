from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import os

# Try import CPETDataLoader from vox_cpet
try:
    from vox_cpet.extractors.base_extractor import CPETDataLoader  # type: ignore
except Exception:
    # Try to add a common workspace path dynamically
    try:
        import sys
        sys.path.append('/home/cheng/workspace/vox_cpet')
        from vox_cpet.extractors.base_extractor import CPETDataLoader  # type: ignore
    except Exception:
        CPETDataLoader = None  # type: ignore


class H5CaseSource:
    """Cache-aware HDF5 case source: loads once, serves per-exam timeseries and summaries."""

    def __init__(self, h5_path: str):
        if not os.path.exists(h5_path):
            raise FileNotFoundError(h5_path)
        if CPETDataLoader is None:
            raise ImportError('vox_cpet not available; cannot read HDF5 source')
        self.h5_path = h5_path
        self._exams: Optional[Dict[str, Dict[str, Any]]] = None
        self._meta: Dict[str, Any] = {}
        self._load()

    def _load(self):
        loader = CPETDataLoader()
        data = loader.load_data(file_type='hdf5', file_path=self.h5_path)
        self._meta = data.get('metadata', {})
        self._exams = data.get('examinations', {})

    def list_exam_ids(self):
        return list((self._exams or {}).keys())

    def get_timeseries(self, exam_id: str) -> pd.DataFrame:
        ex = (self._exams or {}).get(exam_id)
        if not ex:
            raise KeyError(f'exam not found: {exam_id}')
        ts = ex.get('timeseries')
        if ts is None:
            # return empty DataFrame with required columns
            return pd.DataFrame({'Time': []})
        # Ensure Time column exists (schema-transform should do)
        if 'Time' not in ts.columns:
            # Try to find time-like column
            for cand in ['Time', 'time', 'Seconds', 'Second']:
                if cand in ts.columns:
                    ts = ts.rename(columns={cand: 'Time'})
                    break
        return ts

    def get_summary(self, exam_id: str) -> Dict[str, Any]:
        ex = (self._exams or {}).get(exam_id)
        if not ex:
            raise KeyError(f'exam not found: {exam_id}')
        summ = ex.get('summary') or {}
        return dict(summ)

    def get_time_at_at(self, exam_id: str) -> Optional[float]:
        s = self.get_summary(exam_id)
        val = s.get('Time_at_AT')
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return None
        try:
            return float(val)
        except Exception:
            # try parse mm:ss
            try:
                ss = str(val)
                if ':' in ss:
                    parts = [float(p) for p in ss.split(':')]
                    if len(parts) == 3:
                        h, m, sec = parts
                        return h*3600 + m*60 + sec
                    if len(parts) == 2:
                        m, sec = parts
                        return m*60 + sec
            except Exception:
                pass
        return None

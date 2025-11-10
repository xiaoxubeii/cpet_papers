import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

# Minimal SOP: 10s window centered at Time@AT, time-weighted mean with light winsorization.

DEFAULT_WINDOW_S = 10.0
DEFAULT_WINSOR = (0.005, 0.995)


def _parse_time_to_seconds(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, (int, float)) and np.isfinite(x):
            return float(x)
        s = str(x).strip()
        if not s:
            return None
        if ":" in s:
            parts = s.split(":")
            parts = [float(p) for p in parts]
            if len(parts) == 3:
                h, m, sec = parts
                return h * 3600 + m * 60 + sec
            if len(parts) == 2:
                m, sec = parts
                return m * 60 + sec
        return float(s)
    except Exception:
        return None


def parse_time_s(x: Any) -> Optional[float]:
    """Public helper used across annotator components."""
    return _parse_time_to_seconds(x)


def _winsorize(vals: np.ndarray, lower_q: float, upper_q: float) -> np.ndarray:
    mask = np.isfinite(vals)
    if mask.sum() < 3:
        return vals
    finite_vals = vals[mask]
    lq, uq = np.quantile(finite_vals, [lower_q, upper_q])
    if not np.isfinite(lq) or not np.isfinite(uq) or np.isclose(lq, uq):
        return vals
    out = vals.copy()
    out[mask] = np.clip(out[mask], lq, uq)
    return out


def time_weighted_value_at_time(times_s: np.ndarray,
                                values: np.ndarray,
                                t_at: float,
                                window_s: float = DEFAULT_WINDOW_S,
                                winsor: Tuple[float, float] = DEFAULT_WINSOR) -> float:
    if times_s is None or values is None:
        return np.nan
    if len(times_s) != len(values) or len(times_s) < 2:
        return np.nan
    # Ensure ascending
    order = np.argsort(times_s)
    t = np.asarray(times_s, dtype=float)[order]
    v = np.asarray(values, dtype=float)[order]
    if not np.isfinite(t_at):
        return np.nan
    left = t_at - window_s / 2.0
    right = t_at + window_s / 2.0
    mids = (t[:-1] + t[1:]) / 2.0
    lb = np.concatenate(([t[0] - (mids[0] - t[0])], mids))
    rb = np.concatenate((mids, [t[-1] + (t[-1] - mids[-1])]))
    dt = np.maximum(0.0, np.minimum(rb, right) - np.maximum(lb, left))
    mask = dt > 0
    if not np.any(mask):
        return np.nan
    vv = v[mask].astype(float)
    dd = dt[mask].astype(float)
    dd_sum = dd.sum()
    if dd_sum <= 0:
        return np.nan
    dd /= dd_sum
    if winsor and vv.size >= 3:
        vv = _winsorize(vv, winsor[0], winsor[1])
    return float(np.sum(dd * vv))


def derive_values_from_timeseries(ts_df: pd.DataFrame,
                                  time_at_at: Any,
                                  window_s: float = DEFAULT_WINDOW_S) -> Dict[str, Any]:
    """Compute VO2_kg_at_AT, HR_at_AT, Workload_at_AT from timeseries using SOP.
    Required ts_df columns: 'Time' (sec or parsable), and any of 'VO2_kg', 'HR', 'Power_Load'.
    """
    t = ts_df.get('Time')
    if t is None:
        raise ValueError('timeseries missing Time column')
    t_sec = t.apply(_parse_time_to_seconds).to_numpy(dtype=float)
    t_at = _parse_time_to_seconds(time_at_at)

    out: Dict[str, Any] = {
        'VO2_kg_at_AT': np.nan,
        'HR_at_AT': np.nan,
        'Workload_at_AT': np.nan,
        'coverage_ratio': np.nan,
        'time_delta_s': np.nan,
        'warnings': ''
    }

    if not np.isfinite(t_at):
        out['warnings'] = 'invalid_time'
        return out

    # Compute coverage ratio using time bounds overlap
    left = t_at - window_s / 2.0
    right = t_at + window_s / 2.0
    if t_sec.size < 2:
        out['warnings'] = 'few_samples'
        return out
    mids = (t_sec[:-1] + t_sec[1:]) / 2.0
    lb = np.concatenate(([t_sec[0] - (mids[0] - t_sec[0])], mids))
    rb = np.concatenate((mids, [t_sec[-1] + (t_sec[-1] - mids[-1])]))
    dt = np.maximum(0.0, np.minimum(rb, right) - np.maximum(lb, left))
    coverage = float(np.sum(dt)) / float(window_s)
    out['coverage_ratio'] = max(0.0, min(1.0, coverage))

    # VO2/kg
    if 'VO2_kg' in ts_df.columns:
        out['VO2_kg_at_AT'] = time_weighted_value_at_time(t_sec, pd.to_numeric(ts_df['VO2_kg'], errors='coerce').to_numpy(), t_at, window_s)
    elif 'VO2' in ts_df.columns and 'Weight' in ts_df.columns:
        vo2kg = pd.to_numeric(ts_df['VO2'], errors='coerce').to_numpy() / np.maximum(1e-6, pd.to_numeric(ts_df['Weight'], errors='coerce').to_numpy())
        out['VO2_kg_at_AT'] = time_weighted_value_at_time(t_sec, vo2kg, t_at, window_s)
    else:
        out['warnings'] = (out['warnings'] + ';' if out['warnings'] else '') + 'missing_vo2kg'

    # HR
    if 'HR' in ts_df.columns:
        out['HR_at_AT'] = time_weighted_value_at_time(t_sec, pd.to_numeric(ts_df['HR'], errors='coerce').to_numpy(), t_at, window_s)
    # Workload
    if 'Power_Load' in ts_df.columns:
        out['Workload_at_AT'] = time_weighted_value_at_time(t_sec, pd.to_numeric(ts_df['Power_Load'], errors='coerce').to_numpy(), t_at, window_s)

    # time delta is kept as 0 for SOP center; can be extended to expose representative sample time
    out['time_delta_s'] = 0.0

    # warnings
    warns = []
    if out['coverage_ratio'] < 0.8:
        warns.append('low_coverage')
    # borderline flags computed externally where thresholds are known
    if ts_df.shape[0] < 3:
        warns.append('few_samples')
    if warns:
        out['warnings'] = (out['warnings'] + ';' if out['warnings'] else '') + ','.join(warns)

    return out

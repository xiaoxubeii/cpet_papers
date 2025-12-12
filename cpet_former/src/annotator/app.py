"""
CPET AT Annotator (SOP-aligned)
- 双盲两位主读（AT/质量/置信度/可判性），资深仲裁合成共识
- 判读分辨率 10s（c10）；对外报告 30s（c30）
- 自动计算并按规则输出 VO2_AT 与 VO2_peak（30s口径，四舍五入）
Run:
  streamlit run app.py -- --cases <path> --h5 <cpet_data_source.h5> --outdir <outdir> --reader <ID>
"""
from __future__ import annotations
import argparse
import os
import time
import datetime as dt
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from dataclasses import dataclass

try:
    import streamlit as st
except Exception:
    st = None

# Optional: capture Plotly click events for on-chart interaction
try:
    from streamlit_plotly_events import plotly_events  # type: ignore
except Exception:
    plotly_events = None  # type: ignore

# Local imports (works when running inside this folder with Streamlit)
try:
    from data_io import load_cases, append_row_csv  # type: ignore
    from sop_at_value import derive_values_from_timeseries, parse_time_s  # type: ignore
    from h5_backend import H5LazyCaseSource  # type: ignore
except ImportError:
    if __package__:
        from .data_io import load_cases, append_row_csv  # type: ignore
        from .sop_at_value import derive_values_from_timeseries, parse_time_s  # type: ignore
        from .h5_backend import H5LazyCaseSource  # type: ignore
    else:
        raise

THRESHOLDS = [11.0, 14.0]
BASE_TIME_ANCHOR = dt.datetime(2000, 1, 1)
SOP_VERSION = "v1.0"


def seconds_to_anchor_datetime(val: float) -> dt.datetime:
    return BASE_TIME_ANCHOR + dt.timedelta(seconds=float(max(0.0, val)))


def anchor_datetime_to_seconds(value: dt.datetime) -> float:
    return float((value - BASE_TIME_ANCHOR).total_seconds())


def borderline_flags(vo2kg: float, margin: float = 0.3):
    flags = {}
    if not np.isfinite(vo2kg):
        for thr in THRESHOLDS:
            flags[f'borderline_{int(thr)}'] = False
        return flags
    for thr in THRESHOLDS:
        flags[f'borderline_{int(thr)}'] = (abs(vo2kg - thr) <= margin)
    return flags


def _num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors='coerce').to_numpy()


def compute_derived_panels(ts: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Prepare numeric arrays for multiple panels.
    Returns (df_num, info) where df_num has numeric columns and 'Time_s'.
    """
    t = ts['Time'].apply(parse_time_s).to_numpy(dtype=float)
    out = pd.DataFrame({'Time_s': t})
    for col in ['VO2', 'VCO2', 'VE', 'VO2_kg', 'HR', 'Power_Load', 'PetO2', 'PetCO2']:
        if col in ts.columns:
            out[col] = _num(ts[col])
        else:
            out[col] = np.nan
    # derived ratios
    with np.errstate(divide='ignore', invalid='ignore'):
        out['RER'] = np.where(np.isfinite(out['VO2']) & (
            out['VO2'] != 0), out['VCO2']/out['VO2'], np.nan)
        out['VE_per_VO2'] = np.where(np.isfinite(out['VO2']) & (
            out['VO2'] != 0), out['VE']/out['VO2'], np.nan)
        out['VE_per_VCO2'] = np.where(np.isfinite(out['VCO2']) & (
            out['VCO2'] != 0), out['VE']/out['VCO2'], np.nan)
    info = {
        'has_pet': ('PetO2' in ts.columns) or ('PetCO2' in ts.columns),
    }
    return out, info


def infer_sampling_window(ts: pd.DataFrame) -> float:
    """Estimate window length from sampling period; fallback to 10 s."""
    if ts is None or ts.empty or 'Time' not in ts.columns:
        return 10.0
    try:
        t = ts['Time'].apply(parse_time_s).to_numpy(dtype=float)
    except Exception:
        return 10.0
    t = t[np.isfinite(t)]
    if t.size < 2:
        return 10.0
    diffs = np.diff(np.sort(t))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 10.0
    median = float(np.median(diffs))
    # clamp to reasonable bounds
    return float(np.clip(median, 1.0, 30.0))


def round_ml_min_to_10(x: float) -> float:
    if not np.isfinite(x):
        return np.nan
    return float(int(round(x / 10.0)) * 10)


def round_mlkg_min_1dp(x: float) -> float:
    if not np.isfinite(x):
        return np.nan
    return float(np.round(x, 1))


def compute_bins(t_s: float, t0: float = 0.0) -> Tuple[Optional[int], Optional[int]]:
    """Return (c10, c30) where cN = round((t - t0)/N)."""
    if not np.isfinite(t_s):
        return None, None
    c10 = int(round((float(t_s) - float(t0)) / 10.0))
    c30 = int(round((float(t_s) - float(t0)) / 30.0))
    return c10, c30


def estimate_dt_sec(ts: pd.DataFrame) -> float:
    try:
        t = ts['Time'].apply(parse_time_s).to_numpy(dtype=float)
        t = t[np.isfinite(t)]
        if t.size < 3:
            return 1.0
        dif = np.diff(np.sort(t))
        dif = dif[(dif > 0) & np.isfinite(dif)]
        if dif.size == 0:
            return 1.0
        return float(np.median(dif))
    except Exception:
        return 1.0


def rolling_mean(values: np.ndarray, window_n: int, min_periods: Optional[int] = None) -> np.ndarray:
    """Fast rolling mean with NaN handling."""
    if window_n <= 1:
        return values.astype(float)
    s = pd.Series(values, dtype=float)
    return s.rolling(window=window_n, center=True, min_periods=(min_periods or window_n // 2)).mean().to_numpy()


@dataclass
class ReportValues:
    t_at_s: float
    c10_at: Optional[int]
    c30_at: Optional[int]
    vo2_at: float
    vo2_at_kg: float
    vo2_peak: float
    vo2_peak_kg: float
    c30_peak: Optional[int]


def compute_report_values(ts: pd.DataFrame, t_at_s: float) -> ReportValues:
    """SOP reporting values:
    - VO2_AT = time-weighted mean VO2 over 30s centered at c30 anchor
    - VO2_peak = max 30s rolling mean VO2 over full test
    Both also for VO2/kg; rounding applied at the end.
    """
    t = ts['Time'].apply(parse_time_s).to_numpy(dtype=float)
    c10, c30 = compute_bins(t_at_s, t0=0.0)
    # Snap AT to c30*30 for reporting
    t_at_c30 = float(c30 * 30.0) if c30 is not None else float(t_at_s)
    # AT values (30s window)
    # local import to avoid circular hints
    from sop_at_value import time_weighted_value_at_time
    vo2 = pd.to_numeric(ts.get('VO2', pd.Series(
        index=ts.index, dtype=float)), errors='coerce').to_numpy()
    vo2kg = pd.to_numeric(ts.get('VO2_kg', pd.Series(
        index=ts.index, dtype=float)), errors='coerce').to_numpy()
    vo2_at = time_weighted_value_at_time(t, vo2, t_at_c30, window_s=30.0)
    vo2_at_kg = time_weighted_value_at_time(
        t, vo2kg, t_at_c30, window_s=30.0) if 'VO2_kg' in ts.columns else np.nan

    # Peak over 30s rolling mean
    dt_est = estimate_dt_sec(ts)
    nwin = max(1, int(round(30.0 / max(dt_est, 1e-6))))
    nmin = max(1, nwin // 2)
    vo2_roll = rolling_mean(vo2, nwin, min_periods=nmin)
    vo2kg_roll = rolling_mean(
        vo2kg, nwin, min_periods=nmin) if 'VO2_kg' in ts.columns else np.full_like(t, np.nan, dtype=float)
    # Index of peak ignoring NaN (safe)
    if np.any(np.isfinite(vo2_roll)):
        peak_idx = int(np.nanargmax(vo2_roll))
    else:
        peak_idx = 0
    t_peak = float(t[peak_idx]) if peak_idx < len(t) else float('nan')
    _, c30_peak = compute_bins(t_peak, t0=0.0)
    vo2_peak = float(vo2_roll[peak_idx]) if len(vo2_roll) else np.nan
    vo2_peak_kg = float(vo2kg_roll[peak_idx]) if len(vo2kg_roll) else np.nan

    # Rounding rules
    vo2_at_r = round_ml_min_to_10(vo2_at)
    vo2_at_kg_r = round_mlkg_min_1dp(vo2_at_kg)
    vo2_peak_r = round_ml_min_to_10(vo2_peak)
    vo2_peak_kg_r = round_mlkg_min_1dp(vo2_peak_kg)

    return ReportValues(
        t_at_s=float(t_at_s),
        c10_at=c10,
        c30_at=c30,
        vo2_at=vo2_at_r,
        vo2_at_kg=vo2_at_kg_r,
        vo2_peak=vo2_peak_r,
        vo2_peak_kg=vo2_peak_kg_r,
        c30_peak=c30_peak,
    )


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', required=True, help='Path to cases.csv')
    parser.add_argument('--outdir', default='.', help='Output directory')
    parser.add_argument('--h5', required=True,
                        help='HDF5 source file (cpet_data_source_*.h5)')
    parser.add_argument('--reader', default=None, help='Reader ID')
    args = parser.parse_args()

    if st is None:
        raise SystemExit(
            'Streamlit not available. Install streamlit to run UI.')

    return main_app(args.cases, args.h5, args.outdir, args.reader)


def main_app(cases_csv: str, h5_path: str, outdir: str, reader_id_default: Optional[str] = None):
    st.set_page_config(page_title='AT Annotator', layout='wide')

    sidebar = st.sidebar
    sidebar.subheader('统计与配置')
    reader_id = reader_id_default or ''
    assign_path = 'assignment.csv'

    os.makedirs(outdir, exist_ok=True)

    if not cases_csv or not os.path.exists(cases_csv):
        st.error('Provide valid cases.csv path')
        st.stop()

    cases_df = load_cases(cases_csv)
    # H5 source
    try:
        # lightweight lazy HDF5 reader
        h5src = H5LazyCaseSource(h5_path)
    except Exception as e:
        st.error(f'HDF5 load failed: {e}')
        st.stop()

    # optional assignment.csv to filter by reader
    if assign_path and os.path.exists(assign_path):
        asg = pd.read_csv(assign_path)
        if reader_id:
            my_asg = asg.loc[asg['reader_id'] == reader_id].copy()
            # sort by risk desc, then batch, then case_id for stable order
            sort_cols = [c for c in ['risk', 'batch_id']
                         if c in my_asg.columns]
            ascending = [False, True][:len(sort_cols)]
            if sort_cols:
                my_asg = my_asg.sort_values(sort_cols, ascending=ascending)
            keep_ids = my_asg['case_id'].astype(str).tolist()
            cases_df = cases_df[cases_df['case_id'].astype(
                str).isin(set(keep_ids))]
            # reorder cases to follow assignment order (hard priority to high risk)
            cases_df['__order'] = cases_df['case_id'].astype(
                str).map({cid: i for i, cid in enumerate(keep_ids)})
            cases_df = cases_df.sort_values('__order').drop(columns='__order')

    # session state for index
    if 'idx' not in st.session_state:
        st.session_state['idx'] = 0
    idx = st.session_state['idx']

    def move(delta):
        st.session_state['idx'] = int(
            np.clip(idx + delta, 0, max(0, len(cases_df) - 1)))

    if len(cases_df) == 0:
        st.warning('No cases to annotate')
        st.stop()

    case = cases_df.iloc[idx]
    cid = str(case['case_id'])
    st.title(
        f"CPET 用例 {case['case_id']} ({case.get('center', '')}) [{idx+1}/{len(cases_df)}]")

    # Load timeseries
    # cache timeseries loading per case for speed
    @st.cache_data(show_spinner=False, max_entries=256)
    def _cached_ts(h5_path: str, exam_id: str):
        cols = ['VO2', 'VO2_kg', 'VCO2', 'VE', 'HR',
                'Power_Load', 'PetO2', 'PetCO2', 'Time']
        return h5src.get_timeseries(exam_id, columns=cols)

    try:
        ts = _cached_ts(h5_path, str(case['case_id']))
    except Exception as e:
        st.error(f"Failed to load timeseries from HDF5: {e}")
        st.stop()

    # Time@AT handling
    time_at_at_s = parse_time_s(case.get('time_at_at_s'))
    if not np.isfinite(time_at_at_s):
        # fallback to H5 summary if not provided in cases.csv
        time_at_at_s = h5src.get_time_at_at(str(case['case_id']))
    if not np.isfinite(time_at_at_s):
        st.error('Time_at_AT not found (cases.csv or HDF5 summary).')
        st.stop()

    # Controls rendered directly in sidebar
    auto_window = infer_sampling_window(ts)
    window_labels = [
        f'按采样频率 (≈{auto_window:.1f}s)',
        '5 s',
        '10 s',
        '15 s',
        '30 s (报告口径)',
    ]
    window_map = {
        window_labels[0]: auto_window,
        window_labels[1]: 5.0,
        window_labels[2]: 10.0,
        window_labels[3]: 15.0,
        window_labels[4]: 30.0,
    }
    window_choice = sidebar.selectbox(
        '图表平滑窗口', options=window_labels, index=0, help='仅影响图表显示；报告固定 30s')
    window_s = window_map.get(window_choice, auto_window)
    use_vo2kg = sidebar.checkbox('VO2 使用按体重归一', value=('VO2_kg' in ts.columns))

    markdown_separator = sidebar.markdown('---')

    sidebar.subheader('数据标注（SOP）')
    role_label = sidebar.selectbox('角色', options=['主读1', '主读2', '资深仲裁'], index=0)
    role_map = {'主读1': 'r1', '主读2': 'r2', '资深仲裁': 'arbiter'}
    role = role_map[role_label]
    # 可判性、质量、置信度与依据
    readable = sidebar.radio('可判性', options=['可判', '不可判'], index=0) == '可判'
    quality = sidebar.selectbox('质量等级', options=['A', 'B', 'C'], index=0,
                                help='A=完整良好；B=轻度缺陷可判；C=关键缺陷通常不可判')
    confidence = int(sidebar.slider(
        '置信度 (1-5)', min_value=1, max_value=5, value=4))
    rationale = ''
    unreadable_reasons = ''
    if readable:
        rationale = sidebar.text_input('判读依据（可选）', value='')
    else:
        reasons = sidebar.multiselect('不可判定原因', options=[
            '长时掉点', '重漂移', '步进不稳', '努力不足', '同步/校准异常', '数据缺失', '其他'
        ], default=['其他'])
        unreadable_reasons = ';'.join(reasons) if reasons else '其他'
        if quality != 'C':
            sidebar.info('不可判样本通常标记为质量C；若非C请补充说明。')
    # Allow re-annotation: adjust Time@AT via a slider when enabled.
    # We persist per-case edits in session_state using a dict keyed by case_id.
    if 'time_at_at_edits' not in st.session_state:
        st.session_state['time_at_at_edits'] = {}
    slider_coarse_key = f'at_slider_{cid}'
    slider_fine_key = f'at_slider_fine_{cid}'
    slider_pending_key = f'at_slider_pending_{cid}'
    re_annotate = sidebar.checkbox(
        '标注 AT 时间', value=False, help='启用后可直接拖拽图上红线（或用滑块微调）AT 时间（判读分辨率10s）')
    if re_annotate:
        # compute min/max from data for slider bounds
        try:
            t_all = ts['Time'].apply(parse_time_s).to_numpy(dtype=float)
            tmin_s = float(np.nanmin(t_all)) if t_all.size else 0.0
            tmax_s = float(np.nanmax(t_all)) if t_all.size else max(
                60.0, time_at_at_s)
            if not np.isfinite(tmin_s):
                tmin_s = 0.0
            if not np.isfinite(tmax_s) or tmax_s <= tmin_s:
                tmax_s = max(tmin_s + 60.0, time_at_at_s)
        except Exception:
            tmin_s, tmax_s = 0.0, max(60.0, float(
                time_at_at_s) if np.isfinite(time_at_at_s) else 300.0)

        # initial value: previous edit for this case if exists; otherwise current AT
        init_val = st.session_state['time_at_at_edits'].get(
            cid, float(time_at_at_s))
        init_val = float(np.clip(init_val, tmin_s, tmax_s))
        # choose a reasonable step based on sampling window (>=0.5s)
        try:
            step_s = max(0.5, float(infer_sampling_window(ts)))
        except Exception:
            step_s = 1.0

        pending_dt = st.session_state.pop(slider_pending_key, None)

        slider_min = seconds_to_anchor_datetime(tmin_s)
        slider_max = seconds_to_anchor_datetime(tmax_s)
        slider_default = seconds_to_anchor_datetime(init_val)
        slider_format = 'HH:mm:ss'
        if pending_dt is not None:
            st.session_state[slider_coarse_key] = pending_dt
            st.session_state[slider_fine_key] = pending_dt

        # If plotly_events is available, prefer on-chart interaction and offer an optional fine-tune slider
        if plotly_events is None:
            slider_value = st.session_state.get(
                slider_coarse_key, slider_default)
            at_new_dt = sidebar.slider('AT 时间 (mm:ss)', min_value=slider_min, max_value=slider_max,
                                       value=slider_value, step=dt.timedelta(
                                           seconds=float(step_s)),
                                       format=slider_format, key=slider_coarse_key)
            at_new = anchor_datetime_to_seconds(at_new_dt)
            st.session_state['time_at_at_edits'][cid] = float(at_new)
            time_at_at_s = float(at_new)
        else:
            sidebar.caption('提示：在任一时间面板拖拽红线定位 AT；必要时可用下方滑块微调。')
            fine_step = step_s/2.0 if step_s > 0.5 else step_s
            slider_fine_key = f'at_slider_fine_{cid}'
            slider_value = st.session_state.get(
                slider_fine_key, slider_default)
            at_new_dt = sidebar.slider('微调 AT 时间 (mm:ss)', min_value=slider_min, max_value=slider_max,
                                       value=slider_value, step=dt.timedelta(
                                           seconds=float(fine_step)),
                                       format=slider_format, key=slider_fine_key)
            at_new = anchor_datetime_to_seconds(at_new_dt)
            st.session_state['time_at_at_edits'][cid] = float(at_new)
            time_at_at_s = float(at_new)

    markdown_separator = sidebar.markdown('---')

    col_nav1, col_nav2, col_nav3, col_nav4 = sidebar.columns(4)
    if col_nav1.button('上一个'):
        move(-1)
    if col_nav2.button('下一个'):
        move(+1)
    if col_nav3.button('刷新'):
        # Streamlit API changed: prefer st.rerun(); fallback to experimental_rerun
        try:
            if hasattr(st, 'rerun'):
                st.rerun()
            elif hasattr(st, 'experimental_rerun'):
                st.experimental_rerun()
        except Exception:
            pass

    chart_placeholder = st.empty()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    dfp_base, info = compute_derived_panels(ts)

    def _build_time_plot(current_time: float):
        dfp_plot = dfp_base
        titles = (
            'Workload (time)', 'VO2 (time)', 'VCO2 (time)',
            'VE (time)', 'VE/VO2 & VE/VCO2 (time)', 'RER (time)',
            'PetO2 (time)', 'PetCO2 (time)', 'V-slope (VCO2 vs VO2)'
        )
        fig_local = make_subplots(rows=3, cols=3, subplot_titles=titles)

        def _add_line(row, col, x, y, name):
            max_pts = 2000
            if len(x) > max_pts:
                step = int(np.ceil(len(x) / max_pts))
                xs, ys = x[::step], y[::step]
            else:
                xs, ys = x, y
            fig_local.add_trace(go.Scatter(
                x=xs, y=ys, mode='lines', name=name), row=row, col=col)

        if 'Power_Load' in ts.columns:
            _add_line(1, 1, dfp_plot['Time_s'],
                      dfp_plot['Power_Load'], 'Workload')
        if use_vo2kg and 'VO2_kg' in ts.columns:
            _add_line(1, 2, dfp_plot['Time_s'], dfp_plot['VO2_kg'], 'VO2/kg')
        elif 'VO2' in ts.columns:
            _add_line(1, 2, dfp_plot['Time_s'], dfp_plot['VO2'], 'VO2')
        if 'VCO2' in ts.columns:
            _add_line(1, 3, dfp_plot['Time_s'], dfp_plot['VCO2'], 'VCO2')

        if 'VE' in ts.columns:
            _add_line(2, 1, dfp_plot['Time_s'], dfp_plot['VE'], 'VE')
        _add_line(2, 2, dfp_plot['Time_s'], dfp_plot['VE_per_VO2'], 'VE/VO2')
        _add_line(2, 2, dfp_plot['Time_s'], dfp_plot['VE_per_VCO2'], 'VE/VCO2')
        _add_line(2, 3, dfp_plot['Time_s'], dfp_plot['RER'], 'RER')

        if 'PetO2' in ts.columns:
            _add_line(3, 1, dfp_plot['Time_s'], dfp_plot['PetO2'], 'PetO2')
        if 'PetCO2' in ts.columns:
            _add_line(3, 2, dfp_plot['Time_s'], dfp_plot['PetCO2'], 'PetCO2')
        if 'VO2' in ts.columns and 'VCO2' in ts.columns:
            fig_local.add_trace(
                go.Scatter(x=dfp_plot['VO2'], y=dfp_plot['VCO2'], mode='markers', marker=dict(
                    size=4, color='gray'), name='All'),
                row=3, col=3)

        time_panels = [(1, 1), (1, 2), (1, 3), (2, 1),
                       (2, 2), (2, 3), (3, 1), (3, 2)]
        line_shape_indices_local = []
        for (r, c) in time_panels:
            fig_local.add_vline(x=current_time, line_color='red',
                                line_dash='dash', row=r, col=c)
            line_shape_indices_local.append(len(fig_local.layout.shapes) - 1)

        # Force pan-only interactions so drag targets shapes instead of zooming
        fig_local.update_layout(
            height=900,
            margin=dict(l=10, r=10, t=30, b=10),
            dragmode='pan',
            selectdirection='h',
        )

        def _fmt_mmss(sec: float) -> str:
            if not np.isfinite(sec):
                return ''
            sec = max(0.0, float(sec))
            m = int(sec // 60)
            s = int(round(sec - m*60))
            if s == 60:
                m += 1
                s = 0
            return f"{m:02d}:{s:02d}"

        tmin_local = float(np.nanmin(dfp_base['Time_s'])) if len(
            dfp_base['Time_s']) else 0.0
        tmax_local = float(np.nanmax(dfp_base['Time_s'])) if len(
            dfp_base['Time_s']) else max(60.0, tmin_local + 60.0)
        span = max(1.0, tmax_local - tmin_local)
        candidate = [15, 30, 60, 120, 180, 300, 600]
        step = candidate[0]
        for s in candidate:
            if span / s <= 10:
                step = s
                break
            step = s
        import math
        start = math.ceil(tmin_local / step) * step
        ticks = list(np.arange(start, tmax_local + 0.5*step, step))
        ticktext = [_fmt_mmss(v) for v in ticks]
        for (r, c) in time_panels:
            fig_local.update_xaxes(row=r, col=c, tickmode='array',
                                   tickvals=ticks, ticktext=ticktext, title_text='Time (mm:ss)')
        return fig_local, set(line_shape_indices_local)

    max_adjust_iter = 5 if (re_annotate and plotly_events is not None) else 1
    adjust_iter = 0
    final_fig = None
    while True:
        fig, line_shape_idx_set = _build_time_plot(time_at_at_s)
        final_fig = fig
        if re_annotate and plotly_events is not None and line_shape_idx_set:
            def _time_from_relayout(ev_dict):
                for key, val in ev_dict.items():
                    if not isinstance(key, str) or not key.startswith('shapes['):
                        continue
                    try:
                        idx_part = key.split('[', 1)[1]
                        idx = int(idx_part.split(']', 1)[0])
                    except Exception:
                        continue
                    if idx not in line_shape_idx_set:
                        continue
                    if key.endswith('.x0') or key.endswith('.x1'):
                        try:
                            return float(val)
                        except (TypeError, ValueError):
                            continue
                return None

            cfg_drag = {
                'displaylogo': False,
                'edits': {'shapePosition': True},
                'modeBarButtonsToRemove': ['zoom2d', 'lasso2d', 'select2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
            }
            kwargs = dict(
                click_event=False,
                select_event=False,
                hover_event=False,
                relayout_event=True,
                key=f"plt_{case['case_id']}_{adjust_iter}",
                override_height=900,
            )
            used_events = True
            try:
                with chart_placeholder:
                    events = plotly_events(fig, config=cfg_drag, **kwargs)
            except TypeError:
                try:
                    kwargs_no_cfg = dict(kwargs)
                    with chart_placeholder:
                        events = plotly_events(fig, **kwargs_no_cfg)
                except Exception:
                    used_events = False
                    events = None
            except Exception:
                used_events = False
                events = None

            if not used_events:
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                break
            new_time = None
            if events:
                event_list = events if isinstance(events, list) else [events]
                for ev in event_list:
                    if isinstance(ev, dict):
                        new_time = _time_from_relayout(ev)
                        if new_time is not None:
                            break
            if new_time is not None and np.isfinite(new_time):
                tmin_bound = float(np.nanmin(dfp_base['Time_s'])) if len(
                    dfp_base['Time_s']) else 0.0
                tmax_bound = float(np.nanmax(dfp_base['Time_s'])) if len(
                    dfp_base['Time_s']) else max(60.0, tmin_bound + 60.0)
                if tmin_bound <= new_time <= tmax_bound:
                    time_at_at_s = float(new_time)
                    st.session_state['time_at_at_edits'][cid] = float(new_time)
                    st.session_state[slider_pending_key] = seconds_to_anchor_datetime(
                        new_time)
                    adjust_iter += 1
                    if adjust_iter < max_adjust_iter:
                        continue
            break
        else:
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            break

    # Compute SOP-derived values with possibly updated AT time
    # 10s窗口用于参考显示；报告字段均按 30s 计算
    derived = derive_values_from_timeseries(
        ts, time_at_at_s, window_s=float(min(window_s, 10.0)))
    hr_sop = derived['HR_at_AT']
    wl_sop = derived['Workload_at_AT']
    report_vals = compute_report_values(ts, time_at_at_s)

    flags = borderline_flags(report_vals.vo2_at_kg if np.isfinite(
        report_vals.vo2_at_kg) else np.nan)

    def _format_mmss(sec: float) -> str:
        if not np.isfinite(sec):
            return 'NaN'
        sec = max(0.0, float(sec))
        m = int(sec // 60)
        s = int(round(sec - m * 60))
        if s == 60:
            m += 1
            s = 0
        return f"{m:02d}:{s:02d}"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('AT 时间 (c10/c30)',
                f"{_format_mmss(time_at_at_s)}  (c10={report_vals.c10_at}, c30={report_vals.c30_at})")
    col2.metric('VO2_AT / kg',
                f"{report_vals.vo2_at:.0f} / {report_vals.vo2_at_kg:.1f}")
    col3.metric('VO2_peak / kg',
                f"{report_vals.vo2_peak:.0f} / {report_vals.vo2_peak_kg:.1f}")
    col4.metric('HR@AT / Workload@AT', f"{hr_sop:.1f} / {wl_sop:.1f}")

    # Save
    out_reads_r1 = os.path.join(outdir, 'reads_r1.csv')
    out_reads_r2 = os.path.join(outdir, 'reads_r2.csv')
    out_reads_arb = os.path.join(outdir, 'reads_arbiter.csv')
    out_consensus = os.path.join(outdir, 'consensus.csv')

    def _reads_path_for_role(role: str) -> str:
        if role == 'r1':
            return out_reads_r1
        if role == 'r2':
            return out_reads_r2
        return out_reads_arb

    def _latest_by_case(path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
            if 'created_at' in df.columns:
                df['created_at'] = pd.to_datetime(
                    df['created_at'], errors='coerce')
                df = df.sort_values(['case_id', 'created_at']).drop_duplicates(
                    'case_id', keep='last')
            else:
                df = df.drop_duplicates('case_id', keep='last')
            return df
        except Exception:
            return pd.DataFrame()

    def _try_update_consensus(case_id: str):
        """Generate/update consensus for a case based on r1/r2/arbiter following SOP."""
        r1 = _latest_by_case(out_reads_r1)
        r2 = _latest_by_case(out_reads_r2)
        arb = _latest_by_case(out_reads_arb)
        r1_row = r1.loc[r1['case_id'].astype(str) == str(case_id)].tail(1)
        r2_row = r2.loc[r2['case_id'].astype(str) == str(case_id)].tail(1)
        arb_row = arb.loc[arb['case_id'].astype(str) == str(case_id)].tail(1)

        def _row_to_dict(df: pd.DataFrame) -> dict:
            return df.iloc[0].to_dict() if len(df) else {}

        r1d = _row_to_dict(r1_row)
        r2d = _row_to_dict(r2_row)
        arbd = _row_to_dict(arb_row)

        # Decide consensus
        consensus = {}
        source = ''
        readable_both = (r1d.get('readable', 0) == 1) and (
            r2d.get('readable', 0) == 1)
        quality_ok = (r1d.get('quality') in ['A', 'B']) and (
            r2d.get('quality') in ['A', 'B'])
        same_c30 = (r1d.get('c30_at') is not None) and (
            r1d.get('c30_at') == r2d.get('c30_at'))
        if readable_both and quality_ok and same_c30:
            source = 'auto_both_agree'
            c30_at = int(r1d.get('c30_at'))
            t_at_s = float(c30_at * 30.0)
            rep = compute_report_values(ts, t_at_s)  # recompute on c30 anchor
            consensus = {
                'case_id': case_id,
                'sop_version': SOP_VERSION,
                'consensus': 1,
                'source': source,
                't_AT_s': float(t_at_s),
                'c30_AT': int(c30_at),
                'vo2_AT': float(rep.vo2_at),
                'vo2_AT_kg': float(rep.vo2_at_kg),
                'vo2_peak': float(rep.vo2_peak),
                'vo2_peak_kg': float(rep.vo2_peak_kg),
                'c30_peak': int(rep.c30_peak) if rep.c30_peak is not None else '',
                'quality': 'A' if (r1d.get('quality') == 'A' and r2d.get('quality') == 'A') else 'B',
                'r1_id': r1d.get('reader_id', ''),
                'r2_id': r2d.get('reader_id', ''),
                'created_at': dt.datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            }
        elif len(arb_row):
            # Use arbiter decision
            source = 'arbiter'
            c30_at = arbd.get('c30_at')
            if c30_at is not None and str(c30_at) != '':
                t_at_s = float(int(c30_at) * 30.0)
                rep = compute_report_values(ts, t_at_s)
            else:
                t_at_s = float(arbd.get('t_AT_s', np.nan))
                rep = compute_report_values(
                    ts, t_at_s) if np.isfinite(t_at_s) else report_vals
            consensus = {
                'case_id': case_id,
                'sop_version': SOP_VERSION,
                'consensus': int(arbd.get('consensus', 1)),
                'source': source,
                't_AT_s': float(t_at_s) if np.isfinite(t_at_s) else '',
                'c30_AT': int(arbd.get('c30_at')) if arbd.get('c30_at') not in [None, ''] else '',
                'vo2_AT': float(rep.vo2_at) if np.isfinite(rep.vo2_at) else '',
                'vo2_AT_kg': float(rep.vo2_at_kg) if np.isfinite(rep.vo2_at_kg) else '',
                'vo2_peak': float(rep.vo2_peak) if np.isfinite(rep.vo2_peak) else '',
                'vo2_peak_kg': float(rep.vo2_peak_kg) if np.isfinite(rep.vo2_peak_kg) else '',
                'c30_peak': int(rep.c30_peak) if rep.c30_peak is not None else '',
                'quality': arbd.get('quality', ''),
                'r1_id': r1d.get('reader_id', ''),
                'r2_id': r2d.get('reader_id', ''),
                'arbiter_id': arbd.get('reader_id', ''),
                'created_at': dt.datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            }
        else:
            # No consensus yet
            return

        append_row_csv(out_consensus, consensus)

    if sidebar.button('保存并下一个'):
        if not reader_id:
            st.error('Provide Reader ID in sidebar')
        else:
            # Reportability gate: 仅 A/B 且可判 可对外报告
            reportable = int(
                1 if ((readable) and (quality in ['A', 'B'])) else 0)
            c10, c30 = report_vals.c10_at, report_vals.c30_at
            row = {
                'case_id': case['case_id'],
                'reader_id': reader_id,
                'role': role,
                'sop_version': SOP_VERSION,
                'readable': int(1 if readable else 0),
                'unreadable_reason': '' if readable else (unreadable_reasons or ''),
                'quality': quality,
                'confidence': int(confidence),
                't_AT_s': float(time_at_at_s) if np.isfinite(time_at_at_s) else np.nan,
                'c10_at': int(c10) if c10 is not None else '',
                'c30_at': int(c30) if c30 is not None else '',
                'vo2_AT': float(report_vals.vo2_at) if np.isfinite(report_vals.vo2_at) else '',
                'vo2_AT_kg': float(report_vals.vo2_at_kg) if np.isfinite(report_vals.vo2_at_kg) else '',
                'vo2_peak': float(report_vals.vo2_peak) if np.isfinite(report_vals.vo2_peak) else '',
                'vo2_peak_kg': float(report_vals.vo2_peak_kg) if np.isfinite(report_vals.vo2_peak_kg) else '',
                'c30_peak': int(report_vals.c30_peak) if report_vals.c30_peak is not None else '',
                'hr_at_at': float(hr_sop) if np.isfinite(hr_sop) else '',
                'workload_at_at': float(wl_sop) if np.isfinite(wl_sop) else '',
                'warnings': derived.get('warnings', ''),
                'borderline_11': int(flags['borderline_11']),
                'borderline_14': int(flags['borderline_14']),
                'reportable': reportable,
                'rationale': rationale if readable else '',
                'duration_s': 0.0,
                'created_at': dt.datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            }
            start = st.session_state.get('start_ts', None)
            if start is not None:
                row['duration_s'] = float(time.time() - start)
            append_row_csv(_reads_path_for_role(role), row)
            # Attempt to update consensus for this case (auto or arbiter-based)
            try:
                _try_update_consensus(str(case['case_id']))
            except Exception:
                pass
            move(+1)
            st.success('Saved')
            try:
                if hasattr(st, 'rerun'):
                    st.rerun()
                elif hasattr(st, 'experimental_rerun'):
                    st.experimental_rerun()
            except Exception:
                pass

    if 'start_ts' not in st.session_state:
        st.session_state['start_ts'] = time.time()


if __name__ == '__main__':
    main_cli()

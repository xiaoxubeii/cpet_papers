"""
Simple Streamlit annotator for VO2_kg@AT value confirmation using expert Time@AT.
- Loads cases.csv and optional assignment.csv/readers.csv
- For each case: shows VO2_kg over time with Time@AT marker; computes SOP-derived values
- Doctor confirms/adjusts method/window and saves
Run: streamlit run app.py -- --cases <path> --outdir <outdir> --reader <ID>
"""
from __future__ import annotations
import argparse
import os
import time
import datetime as dt
from typing import Optional, Tuple

import numpy as np
import pandas as pd

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
        out['RER'] = np.where(np.isfinite(out['VO2']) & (out['VO2']!=0), out['VCO2']/out['VO2'], np.nan)
        out['VE_per_VO2'] = np.where(np.isfinite(out['VO2']) & (out['VO2']!=0), out['VE']/out['VO2'], np.nan)
        out['VE_per_VCO2'] = np.where(np.isfinite(out['VCO2']) & (out['VCO2']!=0), out['VE']/out['VCO2'], np.nan)
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


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', required=True, help='Path to cases.csv')
    parser.add_argument('--outdir', default='.', help='Output directory')
    parser.add_argument('--h5', required=True, help='HDF5 source file (cpet_data_source_*.h5)')
    parser.add_argument('--reader', default=None, help='Reader ID')
    args = parser.parse_args()

    if st is None:
        raise SystemExit('Streamlit not available. Install streamlit to run UI.')

    return main_app(args.cases, args.h5, args.outdir, args.reader)


def main_app(cases_csv: str, h5_path: str, outdir: str, reader_id_default: Optional[str] = None):
    st.set_page_config(page_title='AT Annotator', layout='wide')

    sidebar = st.sidebar
    sidebar.subheader('统计信息')
    reader_id = reader_id_default or ''
    assign_path = ''

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
            sort_cols = [c for c in ['risk', 'batch_id'] if c in my_asg.columns]
            ascending = [False, True][:len(sort_cols)]
            if sort_cols:
                my_asg = my_asg.sort_values(sort_cols, ascending=ascending)
            keep_ids = my_asg['case_id'].astype(str).tolist()
            cases_df = cases_df[cases_df['case_id'].astype(str).isin(set(keep_ids))]
            # reorder cases to follow assignment order (hard priority to high risk)
            cases_df['__order'] = cases_df['case_id'].astype(str).map({cid: i for i, cid in enumerate(keep_ids)})
            cases_df = cases_df.sort_values('__order').drop(columns='__order')

    # session state for index
    if 'idx' not in st.session_state:
        st.session_state['idx'] = 0
    idx = st.session_state['idx']

    def move(delta):
        st.session_state['idx'] = int(np.clip(idx + delta, 0, max(0, len(cases_df) - 1)))

    if len(cases_df) == 0:
        st.warning('No cases to annotate')
        st.stop()

    case = cases_df.iloc[idx]
    cid = str(case['case_id'])
    st.title(f"CPET 用例 {case['case_id']} ({case.get('center','')}) [{idx+1}/{len(cases_df)}]")

    # Load timeseries
    # cache timeseries loading per case for speed
    @st.cache_data(show_spinner=False, max_entries=256)
    def _cached_ts(h5_path: str, exam_id: str):
        cols = ['VO2','VO2_kg','VCO2','VE','HR','Power_Load','PetO2','PetCO2','Time']
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
    ]
    window_map = {
        window_labels[0]: auto_window,
        window_labels[1]: 5.0,
        window_labels[2]: 10.0,
        window_labels[3]: 15.0,
    }
    window_choice = sidebar.selectbox('聚合窗口', options=window_labels, index=0, help='默认按照采样频率自动设定，可切换 5/10/15 秒固定窗口')
    window_s = window_map.get(window_choice, auto_window)
    # Map localized labels to internal method ids so downstream logic stays consistent
    method_label = sidebar.selectbox('方法', options=['时间加权', '中位数', '最近点'], index=0, help='切换取值方法')
    method_map = {
        '时间加权': 'time_weighted',
        '中位数': 'median',
        '最近点': 'point_value',
    }
    method = method_map.get(method_label, 'time_weighted')
    # Show the AT window band over time-based panels
    show_window = sidebar.checkbox('显示窗灰带', value=True)
    # Optional: apply the same aggregation window to chart lines to refresh visual resolution
    apply_to_charts = sidebar.checkbox('按聚合窗口平滑图表', value=True,
                                       help='对时间序列使用所选窗口进行滚动平均（中心对齐）以刷新图表分辨率')
    use_vo2kg = sidebar.checkbox('VO2 使用按体重归一', value=('VO2_kg' in ts.columns))

    markdown_separator = sidebar.markdown('---')

    sidebar.subheader('数据标注')
    # Allow re-annotation: adjust Time@AT via a slider when enabled.
    # We persist per-case edits in session_state using a dict keyed by case_id.
    if 'time_at_at_edits' not in st.session_state:
        st.session_state['time_at_at_edits'] = {}
    slider_coarse_key = f'at_slider_{cid}'
    slider_fine_key = f'at_slider_fine_{cid}'
    slider_pending_key = f'at_slider_pending_{cid}'
    re_annotate = sidebar.checkbox('重新标注', value=False, help='启用后可直接拖拽图上红线（或用滑块微调）AT 时间')
    if re_annotate:
        # compute min/max from data for slider bounds
        try:
            t_all = ts['Time'].apply(parse_time_s).to_numpy(dtype=float)
            tmin_s = float(np.nanmin(t_all)) if t_all.size else 0.0
            tmax_s = float(np.nanmax(t_all)) if t_all.size else max(60.0, time_at_at_s)
            if not np.isfinite(tmin_s):
                tmin_s = 0.0
            if not np.isfinite(tmax_s) or tmax_s <= tmin_s:
                tmax_s = max(tmin_s + 60.0, time_at_at_s)
        except Exception:
            tmin_s, tmax_s = 0.0, max(60.0, float(time_at_at_s) if np.isfinite(time_at_at_s) else 300.0)

        # initial value: previous edit for this case if exists; otherwise current AT
        init_val = st.session_state['time_at_at_edits'].get(cid, float(time_at_at_s))
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
            slider_value = st.session_state.get(slider_coarse_key, slider_default)
            at_new_dt = sidebar.slider('AT 时间 (mm:ss)', min_value=slider_min, max_value=slider_max,
                                       value=slider_value, step=dt.timedelta(seconds=float(step_s)),
                                       format=slider_format, key=slider_coarse_key)
            at_new = anchor_datetime_to_seconds(at_new_dt)
            st.session_state['time_at_at_edits'][cid] = float(at_new)
            time_at_at_s = float(at_new)
        else:
            sidebar.caption('提示：在任一时间面板拖拽红线定位 AT；必要时可用下方滑块微调。')
            fine_step = step_s/2.0 if step_s > 0.5 else step_s
            slider_fine_key = f'at_slider_fine_{cid}'
            slider_value = st.session_state.get(slider_fine_key, slider_default)
            at_new_dt = sidebar.slider('微调 AT 时间 (mm:ss)', min_value=slider_min, max_value=slider_max,
                                       value=slider_value, step=dt.timedelta(seconds=float(fine_step)),
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

    # Save
    out_reads = os.path.join(outdir, 'reads_human.csv')
    out_reads_with = os.path.join(outdir, 'reads_human_with_values.csv')

    if sidebar.button('保存并下一个'):
        if not reader_id:
            st.error('Provide Reader ID in sidebar')
        else:
            row = {
                'case_id': case['case_id'],
                'reader_id': reader_id,
                'time_at_at_s': float(time_at_at_s) if np.isfinite(time_at_at_s) else np.nan,
                'vo2_kg_at_at': float(vo2_value) if np.isfinite(vo2_value) else np.nan,
                'hr_at_at': float(hr_sop) if np.isfinite(hr_sop) else np.nan,
                'workload_at_at': float(wl_sop) if np.isfinite(wl_sop) else np.nan,
                'method': method_label,
                'window_s': float(window_s),
                'warnings': derived['warnings'],
                'borderline_11': int(flags['borderline_11']),
                'borderline_14': int(flags['borderline_14']),
                'duration_s': 0.0,
                'created_at': dt.datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            }
            # naive duration (per view)
            start = st.session_state.get('start_ts', None)
            if start is not None:
                row['duration_s'] = float(time.time() - start)
            append_row_csv(out_reads, row)
            append_row_csv(out_reads_with, row)
            move(+1)
            st.success('Saved')
            try:
                if hasattr(st, 'rerun'):
                    st.rerun()
                elif hasattr(st, 'experimental_rerun'):
                    st.experimental_rerun()
            except Exception:
                pass

    # Compute SOP-derived values
    derived = derive_values_from_timeseries(ts, time_at_at_s, window_s=float(window_s))
    vo2_sop = derived['VO2_kg_at_AT']
    hr_sop = derived['HR_at_AT']
    wl_sop = derived['Workload_at_AT']

    # Alternative methods
    vo2_value = vo2_sop
    if method == 'median':
        # median within window
        t = ts['Time'].apply(parse_time_s).to_numpy(dtype=float)
        left, right = time_at_at_s - window_s/2.0, time_at_at_s + window_s/2.0
        mask = (t >= left) & (t <= right)
        vals = pd.to_numeric(ts.get('VO2_kg', pd.Series(index=ts.index, dtype=float)), errors='coerce').to_numpy()
        if mask.any():
            vo2_value = float(np.nanmedian(vals[mask]))
    elif method == 'point_value':
        # nearest sample point within window
        t = ts['Time'].apply(parse_time_s).to_numpy(dtype=float)
        idxs = np.where(np.isfinite(t))[0]
        if idxs.size:
            nearest = int(np.argmin(np.abs(t - time_at_at_s)))
            vo2_value = float(pd.to_numeric(ts.get('VO2_kg', pd.Series(index=ts.index, dtype=float)), errors='coerce').fillna(np.nan).to_numpy()[nearest])

    flags = borderline_flags(vo2_value)

    # Display Time@AT as mm:ss
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
    col1.metric('AT 时间', _format_mmss(time_at_at_s))
    col2.metric('VO2/kg@AT', f"{vo2_value:.2f}" if np.isfinite(vo2_value) else 'NaN')
    col3.metric('HR@AT', f"{hr_sop:.1f}" if np.isfinite(hr_sop) else 'NaN')
    col4.metric('Workload@AT', f"{wl_sop:.1f}" if np.isfinite(wl_sop) else 'NaN')

    # Multi-panel plot
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    dfp, info = compute_derived_panels(ts)
    # Optionally smooth/aggregate displayed curves using the selected window
    dfp_plot = dfp
    if apply_to_charts:
        try:
            # approximate sample period to convert seconds to sample count
            dt_est = float(infer_sampling_window(ts)) if np.isfinite(infer_sampling_window(ts)) else 1.0
        except Exception:
            dt_est = 1.0
        nwin = max(1, int(round(float(window_s) / max(1e-6, dt_est))))
        # require at least 3 points for a visible smoothing effect when possible
        nmin = max(1, nwin // 2)
        cols_to_smooth = ['VO2','VO2_kg','VCO2','VE','HR','Power_Load','PetO2','PetCO2','RER','VE_per_VO2','VE_per_VCO2']
        if nwin > 1:
            dfp_plot = dfp.copy()
            for c in cols_to_smooth:
                if c in dfp_plot.columns:
                    ser = pd.to_numeric(dfp_plot[c], errors='coerce')
                    dfp_plot[c] = ser.rolling(window=nwin, center=True, min_periods=nmin).mean()
    # New CPET nine-panel (Wasserman-style)
    titles = (
        'Workload (time)', 'VO2 (time)', 'VCO2 (time)',
        'VE (time)', 'VE/VO2 & VE/VCO2 (time)', 'RER (time)',
        'PetO2 (time)', 'PetCO2 (time)', 'V-slope (VCO2 vs VO2)'
    )
    fig = make_subplots(rows=3, cols=3, subplot_titles=titles)

    # Row 1: Work, VO2/VO2kg, VCO2
    def _add_line(row, col, x, y, name):
        # downsample to ≤ 2000 points for responsiveness
        max_pts = 2000
        if len(x) > max_pts:
            step = int(np.ceil(len(x) / max_pts))
            xs, ys = x[::step], y[::step]
        else:
            xs, ys = x, y
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name=name), row=row, col=col)

    if 'Power_Load' in ts.columns:
        _add_line(1, 1, dfp_plot['Time_s'], dfp_plot['Power_Load'], 'Workload')
    if use_vo2kg and 'VO2_kg' in ts.columns:
        _add_line(1, 2, dfp_plot['Time_s'], dfp_plot['VO2_kg'], 'VO2/kg')
    elif 'VO2' in ts.columns:
        _add_line(1, 2, dfp_plot['Time_s'], dfp_plot['VO2'], 'VO2')
    if 'VCO2' in ts.columns:
        _add_line(1, 3, dfp_plot['Time_s'], dfp_plot['VCO2'], 'VCO2')

    # Row 2: VE, VE equivalents, RER
    if 'VE' in ts.columns:
        _add_line(2, 1, dfp_plot['Time_s'], dfp_plot['VE'], 'VE')
    _add_line(2, 2, dfp_plot['Time_s'], dfp_plot['VE_per_VO2'], 'VE/VO2')
    _add_line(2, 2, dfp_plot['Time_s'], dfp_plot['VE_per_VCO2'], 'VE/VCO2')
    _add_line(2, 3, dfp_plot['Time_s'], dfp_plot['RER'], 'RER')

    # Row 3: PetO2, PetCO2, V-slope
    if 'PetO2' in ts.columns:
        _add_line(3, 1, dfp_plot['Time_s'], dfp_plot['PetO2'], 'PetO2')
    if 'PetCO2' in ts.columns:
        _add_line(3, 2, dfp_plot['Time_s'], dfp_plot['PetCO2'], 'PetCO2')
    if 'VO2' in ts.columns and 'VCO2' in ts.columns:
        mask_win = (dfp_plot['Time_s'] >= time_at_at_s - window_s/2.0) & (dfp_plot['Time_s'] <= time_at_at_s + window_s/2.0)
        fig.add_trace(
            go.Scatter(x=dfp_plot['VO2'], y=dfp_plot['VCO2'], mode='markers', marker=dict(size=4, color='gray'), name='All'),
            row=3, col=3)
        fig.add_trace(
            go.Scatter(x=dfp_plot.loc[mask_win,'VO2'], y=dfp_plot.loc[mask_win,'VCO2'], mode='markers', marker=dict(size=5, color='red'), name='Window'),
            row=3, col=3)

    # Add AT line and window band to time-based panels (exclude V-slope row3,col3)
    time_panels = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2)]
    line_shape_indices = []
    for (r,c) in time_panels:
        if show_window:
            fig.add_vrect(x0=time_at_at_s - window_s/2.0, x1=time_at_at_s + window_s/2.0,
                          fillcolor='LightSkyBlue', opacity=0.2, line_width=0, row=r, col=c)
        fig.add_vline(x=time_at_at_s, line_color='red', line_dash='dash', row=r, col=c)
        shapes = fig.layout.shapes
        if shapes:
            line_shape_indices.append(len(shapes) - 1)

    line_shape_idx_set = set(line_shape_indices)

    fig.update_layout(height=900, margin=dict(l=10, r=10, t=30, b=10))

    # Format time axis as mm:ss for all time-based panels
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

    tmin = float(np.nanmin(dfp['Time_s'])) if len(dfp['Time_s']) else 0.0
    tmax = float(np.nanmax(dfp['Time_s'])) if len(dfp['Time_s']) else 0.0
    span = max(1.0, tmax - tmin)
    # choose step to get about 8-10 ticks
    candidate = [15, 30, 60, 120, 180, 300, 600]
    step = candidate[0]
    for s in candidate:
        if span / s <= 10:
            step = s
            break
        step = s
    # build ticks
    import math
    start = math.ceil(tmin / step) * step
    ticks = list(np.arange(start, tmax + 0.5*step, step))
    ticktext = [_fmt_mmss(v) for v in ticks]
    for (r,c) in time_panels:
        fig.update_xaxes(row=r, col=c, tickmode='array', tickvals=ticks, ticktext=ticktext, title_text='Time (mm:ss)')
    # Enable draggable AT red line via relayout events when possible
    used_events = False
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

        cfg_drag = {'displaylogo': False, 'edits': {'shapePosition': True}}
        kwargs = dict(
            click_event=False,
            select_event=False,
            hover_event=False,
            relayout_event=True,
            key=f"plt_{case['case_id']}",
            override_height=900,
        )
        events = None
        used_events = True
        try:
            events = plotly_events(fig, config=cfg_drag, **kwargs)
        except TypeError:
            try:
                events = plotly_events(fig, **kwargs)
            except Exception:
                used_events = False
        except Exception:
            used_events = False

        if used_events and events:
            event_list = events if isinstance(events, list) else [events]
            new_time = None
            for ev in event_list:
                if isinstance(ev, dict):
                    new_time = _time_from_relayout(ev)
                    if new_time is not None:
                        break
            if new_time is not None and np.isfinite(new_time):
                tmin = float(np.nanmin(dfp['Time_s'])) if len(dfp['Time_s']) else 0.0
                tmax = float(np.nanmax(dfp['Time_s'])) if len(dfp['Time_s']) else 0.0
                if tmin <= new_time <= tmax:
                    cid = str(case['case_id'])
                    st.session_state['time_at_at_edits'][cid] = float(new_time)
                    slider_dt = seconds_to_anchor_datetime(new_time)
                    st.session_state[slider_pending_key] = slider_dt
                    try:
                        if hasattr(st, 'rerun'):
                            st.rerun()
                        elif hasattr(st, 'experimental_rerun'):
                            st.experimental_rerun()
                    except Exception:
                        pass
    if not used_events:
        st.plotly_chart(fig, use_container_width=True)

    

    if 'start_ts' not in st.session_state:
        st.session_state['start_ts'] = time.time()


if __name__ == '__main__':
    main_cli()

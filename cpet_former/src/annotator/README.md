# AT Annotator (src)

What it is
- A minimal, local annotator for VO2/kg@AT values when Time@AT is given.
- Doctors confirm/adjust the SOP-derived value; outputs CSVs ready for training/QA.

Folder
- app.py                # Streamlit UI (single-screen)
- sop_at_value.py       # 10s time-weighted SOP on breath-by-breath time series
- data_io.py            # CSV I/O helpers
- make_assignments.py   # Risk-aware assignment generator (40 cases/reader)
- derive_values_cli.py  # CLI to derive values from reads_human.csv
- templates/            # cases/readers CSV templates
- sample_data/          # tiny demo cases

Quick start (demo)
1) Generate cases.csv from HDF5 (xuhui example):
   python cpet_former/src/annotator/generate_cases_from_h5.py \
     --h5 /home/cheng/workspace/vox_cpet/data/cpet/source/cpet_data_source_xuhui.h5 \
     --center xuhui --out cpet_former/src/annotator/cases.csv

2) Run UI (requires `streamlit`, `plotly`, `pandas`):
   streamlit run cpet_former/src/annotator/app.py -- \
     --cases cpet_former/src/annotator/cases.csv \
     --h5 /home/cheng/workspace/vox_cpet/data/cpet/source/cpet_data_source_xuhui.h5 \
     --outdir cpet_former/src/annotator --reader J1

3) Assignments (per reader 40):
   python cpet_former/src/annotator/make_assignments.py \
     --cases cpet_former/src/annotator/cases.csv \
     --readers cpet_former/src/annotator/templates/readers_template.csv \
     --h5 /home/cheng/workspace/vox_cpet/data/cpet/source/cpet_data_source_xuhui.h5 \
     --out cpet_former/src/annotator/assignment.csv --compute-risk

4) Derive values from reads (CLI alternative to UI):
   python cpet_former/src/annotator/derive_values_cli.py \
     --cases cpet_former/src/annotator/cases.csv \
     --h5 /home/cheng/workspace/vox_cpet/data/cpet/source/cpet_data_source_xuhui.h5 \
     --reads cpet_former/src/annotator/reads_human.csv \
     --out cpet_former/src/annotator/reads_human_with_values.csv

Outputs
- reads_human.csv: case_id, reader_id, time_at_at_s, method, window_s, confidence, quality, notes, created_at, ...
- reads_human_with_values.csv: adds vo2_kg_at_at, hr_at_at, workload_at_at, coverage_ratio, warnings, borderline_11/14
- assignment.csv: case_id, reader_id, center, risk, batch_id

Notes
- Input cases.csv must have: case_id, center, time_at_at_s (optional; will fallback to HDF5 summary).
- Timeseries read directly from the provided HDF5 source per case_id; CSV mode kept only for legacy.
- SOP window default 10 s (±5). Use 5/15 s or median/point methods for noisy or low coverage windows.

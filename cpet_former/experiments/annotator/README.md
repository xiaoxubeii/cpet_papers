# AT Annotator (Streamlit)

- Single-screen, time-only labeling; SOP auto-derives VO2/kg@AT/HR/Workload.
- Use with assignment.csv to filter each reader's workload (40 cases).

Run:
- streamlit run app.py -- --reader <READER_ID>  # or pick in UI
- Output: reads_human.csv (appends) and reads_human_with_values.csv (derived).

Files:
- app.py                  # UI
- sop_at_value.py         # 10s time-weighted SOP for derived values
- make_assignments.py     # risk-aware assignment (40 cases/reader)
- templates/cases_template.csv
- templates/readers_template.csv
- sample_data/case_001.csv

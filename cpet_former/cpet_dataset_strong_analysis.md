# Multi-center CPET Dataset Profiling (cpet_dataset_strong.h5)

Scope: audit /home/cheng/workspace/cpetx_workspace/cpet_former/artifacts/dataset/split/cpet_dataset_strong.h5 so that the data sections in main_clinical.tex can cite concrete, reproducible numbers.

## 0. Pipeline from raw to mixed/kfold files
- Source file. `/home/cheng/workspace/cpetx_workspace/cpet_former/artifacts/dataset/processed/processed_institutes.h5` is built by `vox_cpet.dataset.cpet_processor.CPETDataProcessor` with 10 s aggregation (`aggregation_interval_seconds=10`), time alignment to main load (`align_time_to_t0=True`, `t0_phase_column=Load_Phase`, `t0_phase_value=1`), optional AT repair strategies enabled (`target_strategies` defaults to `raw` and `at_full_repair`), and no ratio clamps by default. The processor interpolates NaNs, applies optional filters, and writes per-institute tables (features/metadata/targets).
- Mixed split. `src/vox_cpet/dataset/cpet_dataset_generator.py:MultiInstituteDataMixer` ingests the processed tables, enforces stratified 0.8/0.1/0.1 splits by center, writes `metadata/institutes`, `metadata/splits_statistics`, and stores feature/metadata/target columns. Standardization scaffolding exists, but `standardization/applied=false` in `cpet_dataset_strong.h5`, meaning values are saved in physical units and scalers must be fit at training time.
- K-fold variants. `/home/cheng/workspace/cpetx_workspace/cpet_former/artifacts/dataset/kfold/` contains 5-fold files generated from the same processed input via `cpetx-data --split-mode kfold` (logs show AT bin cutpoints like 180/300/480 s). Paths are fold-specific H5 files; they keep the same 10 s aggregation and load-aligned timestamps.
- LOCO variants. `/home/cheng/workspace/cpetx_workspace/cpet_former/artifacts/dataset/loco/` holds leave-one-center-out files produced with `--split-mode loco --loco-heldout-institute <center>`, using the same preprocessing and alignment.

### Raw-to-processed retention (shanxi/rizhao/zhongshan)
- Raw source counts (from `vox_cpet/data/cpet/source/cpet_data_source_<center>.h5`, `examinations` keys): Shanxi 8,791; Rizhao 3,784; Zhongshan 1,973.
- Processed counts (after `CPETDataProcessor` aggregation/alignment/interp and QC in `processed_institutes.h5`): Shanxi 8,785; Rizhao 3,618; Zhongshan 1,633.
- Removed exams: Shanxi 6, Rizhao 166, Zhongshan 340. The processor skips exams when feature tables are empty, missing `Examination_ID`, or filtered by QC rules (e.g., severe corruption, failed parsing). Individual reasons are not stored in the H5; to attribute per-exam causes we would need to re-run `CPETDataProcessor.process_data_iter` with verbose logging enabled.

## 1. Data source, governance, and protocols
- Clinical provenance. main_clinical_bak.tex Section 3.1 states that CPET studies from Zhongshan Hospital (Fudan) and two partner centers were collected between Jan 2023 and Dec 2024 under IRB approval (placeholder IRB No. XXX-202X). The Subject_IDs embedded in the HDF5 (e.g., 2023030307, 2025080708) show uploads through mid-2025, so the manuscript needs the actual data-freeze date and final IRB numbers before submission.
- Institutions and devices. Metadata columns (Subject_ID, Institute_Name, Source_Device) cover three primary centers: Shanxi, Rizhao, Zhongshan. Both Shanxi and Rizhao were exported from Ganshorn systems, while Zhongshan used COSMED. Requirement.txt lines 324-332 remind us that other centers (Xuhui, Punan) live in separate files and should be described as external domains, not part of cpet_dataset_strong.h5.
- Exercise protocol. All entries satisfy the inclusion filters in main_clinical_bak.tex (incremental ramp on a cycle ergometer, complete gas exchange, symptom-limited termination). Load-phase histograms confirm that Shanxi heart-failure clinics run shorter ramps (median main-load share 52 percent), Rizhao health-check programs keep long standardized warm-up and recovery blocks, and Zhongshan follows full tertiary protocols with long rest and cooldown. Peak workload heterogeneity (median 70/110/95 W) quantifies the physical style gap that motivates the style encoder in Section 2.
- Ethics and consent. The same paragraph must eventually list the real IRB IDs for all three centers and state that retrospective consent was waived; the placeholder text in the TeX draft should be replaced.

## 2. Cohort composition and eligibility
- After applying the SOP filters (Requirement.txt lines 300-336) we retain 14,036 examinations (11,228 train / 1,403 val / 1,405 test) and 1,087,070 ten-second windows (about 3,020 hours of data). Center-level counts: Shanxi 8,785 (62.6 percent), Rizhao 3,618 (25.8 percent), Zhongshan 1,633 (11.6 percent).
- Tests shorter than 5 minutes survived extraction (118 Shanxi, 3 Rizhao, 0 Zhongshan). They still obey the >=3 minute rule, but this edge case should be noted in the QA paragraph of the manuscript.
- All 70 waveform or derived features listed in cpetformat/cpet.yaml are present; eight metadata fields (Subject_ID, Gender, Age, Height, Weight, Examination_ID, Source_Device, Institute_Name) drive the stratified sampling.

### Table 1. Demographics and anthropometrics
| Center | Exams | Age mean +/- SD (yr) | Male (%) | BMI mean +/- SD (kg/m^2) | BSA mean +/- SD (m^2) | Device |
|---|---|---|---|---|---|---|
| Shanxi | 8,785 | 58.95 +/- 10.30 | 59.5 | 25.37 +/- 3.42 | 1.80 +/- 0.19 | Ganshorn |
| Rizhao | 3,618 | 61.38 +/- 12.03 | 59.9 | 25.64 +/- 3.36 | 1.81 +/- 0.19 | Ganshorn |
| Zhongshan | 1,633 | 50.58 +/- 14.35 | 72.0 | 23.89 +/- 3.28 | 1.79 +/- 0.20 | COSMED |

Gender encoding follows the extractor logic in vox_cpet/vox_cpet/extractors/base_extractor.py (0 = female, 1 = male) and should be stated explicitly in the paper.

### Table 2. Operational heterogeneity (time-series metadata)
| Center | Median duration (sec / min) | Phase mix (% preload / main / post) | Median preload (sec) | Median postload (sec) | Median max load (W) | Tests <5 min |
|---|---|---|---|---|---|---|
| Shanxi | 600 sec / 10.0 min | 15.5 / 52.4 / 32.1 | 100 | 190 | 70 | 118 |
| Rizhao | 1,070 sec / 17.8 min | 25.8 / 40.3 / 33.9 | 270 | 360 | 110 | 3 |
| Zhongshan | 1,080 sec / 18.0 min | 33.1 / 38.9 / 28.0 | 360 | 300 | 95 | 0 |

## 3. Signal acquisition and preprocessing SOP
- Temporal aggregation. features/Time increments by 10 seconds; this is the meso-scale aggregation described in Methodology 2.1 to suppress hyperventilation noise.
- Phase truncation. Load_Phase codes 0/1/2 (pre/main/post) confirm we trim rest and recovery segments. Median preload durations are center-dependent (Zhongshan still exports the full 6-minute warm-up), so mention this nuance when describing the SOP.
- Time alignment. Dataset generation logs (e.g., vox_cpet/logs/cpetx-data-split-20251122_234337.log) show that we called cpetx-data with --align-time-to-t0 and --t0-phase-column Load_Phase so that Time and Time_at_AT reset at main-load onset. This is important for explaining online inference windows.
- Normalization. metadata/standardization/applied = false and total_standardized_columns = 73, so the split file stores physical units only. The manuscript should state that z-scoring is re-fit on the training split and re-used on val/test.
- Feature catalog. 70 channels (ventilation, gas exchange, hemodynamics, energy, 12-lead ST/S voltages) follow CPET_Standard_v1.4 in vox_cpet/cpetformat/cpet.yaml; referencing that schema avoids listing every column inline.

## 4. Reference standard and label distribution
- Annotation workflow. main_clinical.tex already mentions double-blind consensus with roughly +/- 30 second variability. metadata/target_strategy_stats/raw_missing is empty, so every study has a usable AT timestamp.
- Distribution. targets/raw/Time_at_AT stores 10-second bins. Overall median is 450 seconds (IQR 360-570). Center-specific medians: Shanxi 390 seconds, Rizhao 570 seconds, Zhongshan 621 seconds. These numbers explain why online causal masks are crucial for Shanxi while offline review benefits from the long tail at Zhongshan.

## 5. Split strategy and evaluation protocols
- Mixed-center split. The HDF5 follows a stratified 0.8 / 0.1 / 0.1 split per center (`metadata/splits_statistics`). This is the dataset used for the mixed-center training/validation/offline metrics reported in the paper.
- LOCO variants. Hold-out files live under artifacts/dataset/loco/ (e.g., cpet_dataset_shanxi.h5, cpet_dataset_zhongshan.h5). Generation logs (vox_cpet/logs/cpetx-data-split-20251202_224621.log, etc.) show that we ran `cpetx-data --split-mode loco --loco-heldout-institute <center>` with 90/10 splits on the remaining centers. The evaluation section should state that Early Trigger Rate and VO2@75% metrics on the held-out center are computed with the causal attention mask enabled, per Method Section 2.4.
- External domains. Requirement.txt distinguishes between the primary three-center dataset and external centers (Xuhui, Punan). Those live in other processed files and should be cited separately when discussing unknown-center generalization.

## 6. Missingness and QA
- Numeric completeness. Automated scans across all 70 numeric features (train/val/test) found 0 NaN and 0 Inf entries. BP and ST channels still contain the sparse zero values emitted by the device; if we add masks later, mention it.
- AT coverage. metadata/target_strategy_stats/unique_examinations = 14,036 and raw_missing/examinations = []. No AT labels are missing.
- Edge cases. Short-duration (<5 min) Shanxi tests and very long Rizhao sessions (up to 59.8 min) are the main QC flags. Confirm in the manuscript that clinicians reviewed them before modeling.
- Window volume. 1,087,070 ten-second windows (~3,020 h) yield enough negative/positive slices to measure an Early Trigger Rate below 2 percent with narrow confidence intervals. Mention this when motivating the two deployment modes.

## 7. Action items for the manuscript
1. Update the Study Design paragraph with the real data-freeze date (Subject_IDs show 2025) and final IRB numbers.
2. Insert a dataset table (based on Table 1 and 2 above) so reviewers see sample counts, demographics, devices, and protocol differences.
3. Document both the stratified 0.8/0.1/0.1 split and the LOCO derivatives, pointing to the on-disk file names for reproducibility.
4. In the preprocessing subsection, cite the actual cpetx-data command-line flags (align-to-T0, meso-scale aggregation, preload/recovery truncation) and state that standardization is fit only on the training split.
5. Clarify in the label paragraph that Time_at_AT is binned to 10 seconds, fully observed, and distributed very differently across centers (390 vs. 570 vs. 621 seconds).
6. Note that short tests were reviewed manually and that online metrics (Early Trigger Rate) are computed with the causal mask applied to historical slices only.

All diagnostics were generated by /tmp/analyze_dataset.py; consider committing that helper script (or its key outputs) for supplementary material.

# Repository Guidelines

## Project Structure & Module Organization
The manuscript lives in `main.tex`, which owns the preamble, authorship block, and section order. Keep new packages and macros in the preamble comments for quick review. Place figures inside `figures/` with descriptive snake_case names such as `fig_at_overview.pdf`, and reference them through relative paths (`\includegraphics{figures/fig_at_overview}`). The evolving outline and reviewer notes sit in `requirement.txt`; update it whenever you adjust structure or emphasis so planning stays current.

## Build, Test, and Development Commands
Run `latexmk -pdf main.tex` for a full compile; it automatically performs the extra runs needed for references and hyperlinks. During iterative drafting, `latexmk -pvc -pdf main.tex` watches the sources and rebuilds on save. Clean auxiliary artifacts with `latexmk -c` before you commit. If TeX Live is missing, fall back to `tectonic main.tex`, then compare the PDF to the latexmk output.

## Coding Style & Naming Conventions
Match the light indentation already in `main.tex`: align option lists with two spaces and keep one command per line in complex environments. Group related macros with `%%` separators, and add `%` comments only when intent is not obvious. Title sections in Title Case (`\section{Reader Study Results}`) and label floats with `fig:` or `tab:` prefixes (for example, `\label{fig:at_comparison}`). Prefer `\textit{}` for emphasis and reserve inline math in headings for essential symbols.

## Testing Guidelines
Treat a warning-free `latexmk -pdf main.tex` run as the acceptance test. Review `main.log` for overfull or underfull boxes and fix them before merging. When adding macros or packages, run `chktex main.tex` (part of most TeX distributions) to catch common LaTeX antipatterns. Use a visual diff tool such as `diffpdf` when modifying figures to confirm layout stability.

## Commit & Pull Request Guidelines
Write commit subjects in imperative mood with a short prefix (`docs:`, `fig:`, `infra:`) and stay under 65 characters (for example, `docs: revise reader study narrative`). Add a body when the rationale is not obvious from the diff. Pull requests should summarize the manuscript sections touched, list refreshed figures, and link the corresponding entry in `requirement.txt`. Attach the latest PDF and mention outstanding warnings so reviewers can reproduce the build.

## Figure & Asset Management
Store raw plots, scripts, and provenance alongside each exported figure inside `figures/` subfolders (for example, `figures/at_curve/src/`). Document preprocessing steps in a local README so replacements remain reproducible. Update captions in `main.tex` whenever assets change, and prefer `\includegraphics[width=\linewidth]` to maintain consistent sizing.

## CPET 数据标准和数据集
CPET 数据标准：/root/autodl-tmp/vox_cpet/cpetformat/cpet.yaml
cosmed 数据转换标准：/root/autodl-tmp/vox_cpet/cpetformat/cosmed.yaml
ganshorn 数据转换标准：/root/autodl-tmp/vox_cpet/cpetformat/ganshorn.yaml

数据清理流程：
- 清洗工具：/root/autodl-tmp/vox_cpet/vox_cpet/cmd/cpetx-data
- 清洗流程：/root/autodl-tmp/vox_cpet/vox_cpet/dataset/cpet_processor.py
- 标准化和分割流程：/root/autodl-tmp/vox_cpet/vox_cpet/dataset/cpet_dataset_generator.py

数据集目录在：/root/autodl-tmp/cpet_workspace/artifacts/cpet_dataset
- cpet_dataset：全量数据集
- *_small：10% 数据集
- *_ssl：预训练数据集
- *_explicit_split：train/val 包含三个中心，test 使用了一个外部 punan 数据集
- *_loco：三个数据集交替做 LOCO

## 专家标注 SOP

- 口径冻结：统一 t0、10s 判读分辨率；报告主指标用 30s 分箱（c30=round((tAT-t0)/30)）、辅指标用 10s 分箱；允许“不可判定”并列出理由清单。
- 双盲标注：两名专家独立判读（AT/RCP/限制类型/质量等级/置信度），不可互见；完成先导 20 例（≥10 Ganshorn）达标后再全量。
- 上锁与一致性：先导集出齐即“上锁”，计算加权 κ 与一致率
  - 30s 分箱加权 κ（二次权重）≥0.80 为门槛；10s 分箱 κ ≥0.75 为辅；±10/20/30 s 一致率与 MAE 同时报；按设备分层。
  - 未达标先回看分歧样例、补充判读规则与面板，再抽样复测。
- 共识与仲裁：生成分歧清单 → 第三名资深仲裁；记录裁决与依据；仍不确定标“不可判定（原因）”。
- 版本与存档：保留 r1.json、r2.json、consensus.json 三套；标注版本 v1.0；落盘到 artifacts/external_center_xxx/，并在 requirement.txt 登记规则与阈值。
- 使用原则：外部评测只用共识强标签；个人版仅用于 IRR 与质控；训练集按既定“强/弱标签入组”策略执行。外加 vo2_AT 和 vo2_peak 判读
- vo2_AT/vo2_peak 判读：
   - vo2_AT：以共识 t_AT 对应的 30s 分箱 c30 的 VO2 为主值（mL/min），同步给出体重归一化值（mL/kg/min）；10s 版本仅用于 IRR 统计；若 AT 不可判则标“不可判定（原因）”。
   - vo2_peak：以全程 30s 滚动平均 VO2 的最大值为主值；同时记录发生时刻的 c30；异常孤立尖峰按既定数据清洗流程处理；数据缺失/漂移严重则标“不可判定（原因）”。
   - 取整与报告：主值四舍五入至最近 10 mL/min；mL/kg/min 保留 1 位小数；JSON 字段建议包含：vo2_AT、vo2_AT_kg、vo2_peak、vo2_peak_kg、t_AT、c30_AT、c30_peak、quality、rationale。
   - 一致性与统计：以 30s 分箱评估一致率与 MAE（具体阈值登记在 requirement.txt）；10s 分箱统计作为辅参考；按设备分层报告。

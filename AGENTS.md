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

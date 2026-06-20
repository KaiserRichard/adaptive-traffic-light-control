# ATLC XeLaTeX Report

This folder contains the Vietnamese XeLaTeX report for the ATLC microprocessor-course report.

The report intentionally has no cover page. The expected workflow is:

1. Design the cover separately in Word.
2. Compile this XeLaTeX folder for the report body.
3. Merge the Word cover PDF and this report PDF if needed.

## Structure

```text
reports/thesis_latex/
├── main.tex
├── chapters/
├── figures/
├── tables/
├── references.bib
├── Makefile
└── README.md
```

## Build Locally

Lightweight option already available on this machine:

```bash
cd reports/thesis_latex
tectonic main.tex
```

Traditional XeLaTeX/BibTeX workflow:

```bash
cd reports/thesis_latex
make
```

Manual equivalent:

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

## Overleaf

Upload the whole `reports/thesis_latex` folder to Overleaf and set the compiler to XeLaTeX.

## Source Basis

System-code basis: commit `857423f` from the Phase 15 FreeRTOS controller branch.

This report folder should not modify `pc_app/` or `firmware/`.

# ATLC Evidence Appendix

This folder is a standalone XeLaTeX appendix/evidence pack for the ATLC report.

Use it when the main report should stay lightweight, but you still want a separate PDF containing:

- run commands
- hardware test procedure
- serial/UART traces
- runtime plots
- benchmark metrics
- placeholder figures for future hardware evidence

Compile independently:

```bash
cd reports/thesis_appendix_latex
tectonic main.tex
```

On Overleaf, upload this folder as a separate project or compile it separately from `reports/thesis_latex/`.


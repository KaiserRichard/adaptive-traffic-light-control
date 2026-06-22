# Code-drawn diagram sources

This folder stores editable, code-first sources for report figures. These files are meant to replace AI-generated-looking diagrams with reproducible technical drawings.

Recommended workflow:

1. Edit the `.mmd` file when the diagram structure changes.
2. Paste the Mermaid code into <https://mermaid.live/> or draw.io using `Arrange > Insert > Mermaid`.
3. Export SVG/PDF for Overleaf.
4. Keep the `.mmd` source committed so the diagram remains reproducible.

Current sources:

- `atlc_five_stage_pipeline.mmd`: five-stage ATLC runtime flow.
- `freertos_status_reporting_flow.mmd`: ESP32 FreeRTOS STATUS reporting flow.
- `atlc_five_stage_pipeline.svg`: printable vector version.
- `freertos_status_reporting_flow.svg`: printable vector version.


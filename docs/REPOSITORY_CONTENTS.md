# What belongs in this repository

This clean repository was assembled from the supplied project without copying its task histories, saved databases or development-report archive.

## Included

- All Python source in `core/` and `ui/`: preserves subsystem imports and optional features.
- GUI and CLI entry points, plus read-only accessibility and optional provider diagnostics.
- Planner/executor text prompts, local Markdown task skills, and the three shipped plugin manifests: these are runtime assets rather than generated data.
- A portable configuration example: local Ollama by default, relative data paths and no API keys. Cloud model IDs are left for the user to select.
- Core/full dependency lists and a convenience PowerShell launcher.
- Python test sources, plus the two benchmark utility modules they import.
- One documented dashboard image, a detailed README, architecture notes and the packaging validation record.

## Excluded

| Excluded material | Reason |
| --- | --- |
| Local `config.json`, environment files and credentials | Machine/user-specific; can contain secrets. |
| Task history, memory, file indexes, audit databases and journal files | Private runtime data; created locally when needed. |
| Benchmark result folders, JSON task traces and browser/download fixtures generated during old runs | Bulky development evidence that can contain task/user content. |
| Plugin knowledge documents and generated indexes | User-provided data rather than shipped app manifests. |
| `__pycache__`, virtual environments, downloaded models and browser binaries | Generated/installable dependencies. |
| Archives, videos, progress PDFs/DOCX reports and handoff renderings | Not required to execute the app. |
| Old status documents, changelogs and historical quickstarts | Replaced by a coherent README for this clean package. |
| Other projects' images and documents | Unrelated to PunkOS. |

The original working project is preserved separately. Development benchmark utilities beyond those imported by the included tests are also omitted; their historical reports and results are not represented as reproducible measurements in this package.

## Uploading future changes

Run `git status --short` and `git diff --cached --stat` before committing. Keep `config.example.json` credential-free. Do not force-add ignored databases, local configuration, task logs or knowledge documents. Add further screenshots only after reviewing their visible contents.

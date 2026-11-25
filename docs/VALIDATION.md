# Packaging validation

Validation date: **25 November 2026**. Checks were performed on Windows with Python **3.11.9** against this curated repository, rather than the original development folder. A fresh `.venv` was created and `requirements-core.txt` installed successfully before the final checks below.

## Automated checks

| Check | Observed result |
| --- | --- |
| `python -m pip check` | **No broken requirements found** in the fresh core environment. |
| `python -m unittest discover -s tests -q` | **510 tests passed**, 29.992 seconds reported by unittest in the fresh core environment. |
| `python tests/test_integration.py` | **7 checks passed**, 0.938 seconds including process startup. |
| `python tests/test_modules.py` | **20 checks passed**, 5.828 seconds including process startup, after updating stale fixtures and isolating fallback-embedding tests. |
| `python tests/test_ui.py` | **14 checks passed**, 2.313 seconds including process startup; includes dashboard/panel construction and theme checks. |
| Python syntax | All 129 shipped Python files parsed successfully. |
| Staged file review | Application source, runtime assets, tests, documentation and one screenshot only; no local config, databases, virtual environment, task recordings or report archives. |
| Common credential-pattern and machine-path scan | No matches for the checked real-key patterns or original machine-specific user/storage paths in shipped text. This is a targeted check, not a comprehensive secret-detection guarantee. |

The 41 standalone checks are listed separately because their function-based runners are not collected by `unittest discover`. They are not 41 additional real user-task trials. Logged HTTP retry messages in the regression suite come from simulated provider cases.

Key packages resolved during the fresh installation included Ollama 0.6.3, PyQt6 6.12.1, Playwright 1.65.2, HTTPX 0.28.1, NumPy 2.4.6 and jsonschema 4.26.0. Dependency files use minimum versions rather than a complete environment lock; later resolutions can differ. Browser checks exercised the available local browser setup with headless fixtures. A new Chromium binary download was not performed during this check; the README includes the install command for new machines.

## Test-fixture maintenance

One cloud checker test previously depended on a nonempty Gemini model in the user's actual configuration. It now supplies a synthetic model configuration explicitly, so it checks the no-API-call behavior independently of local settings.

Two old learning fixtures claimed successful input without the completion-check receipt now required by the implementation. They now include a synthetic successful completion check. The fallback-embedding and memory/learning checks explicitly use the lexical backend so running these unit checks does not load or download a transformer model. Production agent behavior was not relaxed to make the checks pass.

## Scope

These checks validate packaging, regressions and selected local fixture/UI behavior. They do not establish general desktop-task accuracy, model quality, safe arbitrary automation, or typical latency. The showcase image is an offline replay of a historical result and is not evidence of a newly executed task.

No live cloud inference, real desktop input or task execution was performed for this packaging validation. The configured local `qwen2.5:3b` model must be downloaded separately. Full optional voice and semantic-model installation is not included in the smaller core setup check.

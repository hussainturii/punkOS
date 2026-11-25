# Architecture notes

PunkOS uses a provider-independent orchestration layer. GUI and CLI entry points construct shared subsystems, while the dashboard executes agent work on a worker thread.

| Area | Main files | Responsibility |
| --- | --- | --- |
| Entry points | `main.py`, `run_cli.py`, `core/bootstrap.py` | Load configuration and assemble subsystems. |
| Orchestration | `core/planner.py`, `core/router.py`, `core/reasoning.py` | Route goals, plan work, choose actions and bound recovery. |
| Desktop grounding | `core/perception.py`, `core/window_handoff.py` | Read accessible controls and identify relevant application windows. |
| Input | `core/controller.py`, `core/input_evidence.py` | Execute desktop operations and distinguish input receipts from observed results. |
| Browser | `core/web_agent.py`, `core/web_tasks.py` | Playwright session control, DOM observations and task execution. |
| Files | `core/file_intel.py`, `core/file_operations.py`, `core/embedding.py` | Extraction, indexing, retrieval and structured operations. |
| Providers | `core/llm.py`, `core/cloud_llm.py`, `core/inference_transport.py` | Ollama/cloud clients, structured outputs and request handling. |
| Safety | `core/safety.py`, `core/sandbox.py`, `core/audit.py` | Risk assessment, confirmation, execution checks and audit records. |
| Evidence | `core/agent_checks.py`, `core/requirements.py`, `core/task_report.py` | Check requirements and format honest completion/partial results. |
| Persistence | `core/history.py`, `core/memory.py`, `core/learning.py`, `core/metrics.py` | Task records, procedural knowledge and aggregate measurements. |
| UI | `ui/dashboard.py`, other `ui/` panels | Conversation, plan/result display, browser tools, search, history, plugins and settings. |

## Observations and action loop

Desktop reasoning primarily receives text derived from Windows UI Automation. Where supported and enabled, a provider may also receive images. Browser reasoning uses DOM-derived observations. Observed element identity matters: a successful mouse/keyboard call only proves that input was sent, not that the app accepted the intended change.

The planner collects observations, requests an action, checks risk, requests confirmation where required, executes and records the result. Completion proposals are checked against observed evidence and task requirements. Failure recovery is bounded. Browser, desktop and file tasks use different tools under the shared planner; unrestricted handoff among every application and task type is not established.

## Local data

The example configuration places history and audit databases under `data/history/`, memory under `data/memory/`, and downloads under `data/downloads/`. Plugin knowledge indexes may be generated under `data/plugins/`. Git ignores runtime databases globally and excludes private plugin knowledge folders. Shipped prompts, skills and plugin manifests remain tracked.

## Boundaries

Risk assessment, path checks and confirmations are useful safeguards; arbitrary desktop input is not fully isolated by them. The subprocess runner is not a complete security sandbox. Browser observation and completion checks can miss semantic mistakes. Learned procedures are not a guarantee of correct future behavior; memory recall is disabled in the example configuration. Validate outcomes in the actual target app before relying on a workflow.

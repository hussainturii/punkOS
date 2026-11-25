"""
Subsystem construction, shared by the CLI and the GUI.

Both entry points need the same eleven modules wired together in the same
order. Doing that twice guarantees they drift, so it happens here once.

Every optional subsystem is built defensively: if Playwright is missing, or
sentence-transformers was never installed, or the plugins directory does not
exist, that subsystem comes back as None and PunkOS runs without it rather than
refusing to start. The planner already treats every optional subsystem as
nullable, so a partial install degrades instead of failing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class Subsystems:
    """Everything the planner may optionally use."""
    memory: Any = None
    learner: Any = None
    plugins: Any = None
    sandbox: Any = None
    web_agent: Any = None
    audit_log: Any = None
    file_search: Any = None
    warnings: list[str] = None      # type: ignore[assignment]

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    def as_planner_kwargs(self) -> dict:
        return {
            "memory": self.memory,
            "learner": self.learner,
            "plugins": self.plugins,
            "sandbox": self.sandbox,
            "web_agent": self.web_agent,
            "audit_log": self.audit_log,
            "file_search": self.file_search,
        }


def _resolve(base: Path, value: str) -> Path:
    """Resolve a config path relative to the project root unless absolute."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path)


def build_subsystems(
    config: dict,
    project_root: Optional[Path] = None,
    on_log: Optional[Callable[[str], None]] = None,
) -> Subsystems:
    """Construct every optional subsystem from config.json."""
    root = project_root or Path(__file__).resolve().parent.parent
    log = on_log or (lambda _m: None)
    subs = Subsystems()

    safety_cfg = config.get("safety", {})
    authorized = safety_cfg.get("authorized_directories", [])

    # ---- M9 audit log (built first: other subsystems log into it) ----
    try:
        from .audit import AuditLog
        audit_cfg = config.get("audit", {})
        subs.audit_log = AuditLog(
            _resolve(root, audit_cfg.get("db_path", "data/history/audit.db")),
            retention_days=audit_cfg.get("retention_days", 90),
        )
    except Exception as e:
        subs.warnings.append(f"Audit log unavailable: {e}")

    # ---- M11 procedural memory + M10 learning ----
    try:
        from .memory import ProceduralMemory
        from .learning import ContinuousLearner
        mem_cfg = config.get("memory", {})
        subs.memory = ProceduralMemory(
            _resolve(root, mem_cfg.get("db_path", "data/memory/memory.db"))
        )
        subs.learner = ContinuousLearner(
            subs.memory,
            similarity_threshold=mem_cfg.get("similarity_threshold", 0.85),
            on_log=log,
        )
        # is_semantic loads the transformer. Checking it here blocks the GUI
        # before window.show(), even when memory recall is disabled.
        log('Memory embedding backend will load on first use.')
    except Exception as e:
        subs.warnings.append(f"Memory/learning unavailable: {e}")

    # ---- M8 plugins ----
    try:
        from .plugins import PluginRegistry
        plug_cfg = config.get("plugins", {})
        subs.plugins = PluginRegistry(
            _resolve(root, plug_cfg.get("directory", "data/plugins")),
            require_validation=plug_cfg.get("require_validation", True),
            on_log=log,
            audit_log=subs.audit_log,
        )
        count = subs.plugins.discover()
        from .file_intel import FileSearchEngine
        subs.plugins.knowledge_engine = FileSearchEngine(subs.plugins.directory / 'knowledge-index.db')
        log(f"[bootstrap] {count} plugin(s) loaded")
        if subs.plugins.rejected:
            subs.warnings.append(
                f"{len(subs.plugins.rejected)} plugin(s) failed validation: "
                f"{', '.join(subs.plugins.rejected)}"
            )
    except Exception as e:
        subs.warnings.append(f"Plugin registry unavailable: {e}")

    # ---- M9 sandbox ----
    try:
        from .sandbox import SandboxExecutor
        sb_cfg = config.get("sandbox", {})
        if sb_cfg.get("enabled", True):
            subs.sandbox = SandboxExecutor(
                authorized_dirs=authorized,
                backend=sb_cfg.get("backend", "subprocess"),
                docker_image=sb_cfg.get("docker_image", "python:3.11-slim"),
                timeout_seconds=sb_cfg.get("timeout_seconds", 30),
                memory_limit_mb=sb_cfg.get("memory_limit_mb", 256),
                cpu_limit=sb_cfg.get("cpu_limit", 0.5),
                audit_log=subs.audit_log,
            )
    except Exception as e:
        subs.warnings.append(f"Sandbox unavailable: {e}")

    # ---- M7 web agent (constructed lazily; the browser only starts on demand) ----
    try:
        from .web_agent import WebAgent
        web_cfg = config.get("web", {})
        if web_cfg.get("enabled", True):
            subs.web_agent = WebAgent(
                browser=web_cfg.get("browser", "chromium"),
                headless=web_cfg.get("headless", False),
                max_tabs=web_cfg.get("max_tabs", 5),
                navigation_timeout_ms=web_cfg.get("navigation_timeout_ms", 30000),
                channel=web_cfg.get('channel'),
                download_dir=web_cfg.get('download_dir'),
                on_log=log,
                audit_log=subs.audit_log,
            )
    except Exception as e:
        subs.warnings.append(f"Web agent unavailable: {e}")

    # ---- M6 file search shared with the read-only agent route ----
    try:
        from .file_intel import FileSearchEngine
        subs.file_search = FileSearchEngine(
            _resolve(root, config.get('history', {}).get('db_path', 'data/history/punkos.db')).parent / 'file_index.db'
        )
    except Exception as e:
        subs.warnings.append(f"File search unavailable: {e}")

    return subs

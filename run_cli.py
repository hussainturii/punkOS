"""
PunkOS — CLI entry point.
Runs a single task without launching the GUI. Useful for testing and debugging.

Usage:
    python run_cli.py "Open Calculator and click 7"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.perception import VisualPerceptionEngine
from core.reasoning import ReasoningCore
from core.controller import DesktopController
from core.safety import SafetySystem
from core.history import TaskHistoryStore
from core.planner import StrategicPlanner
from core.llm import build_provider
from core.bootstrap import build_subsystems


def load_config() -> dict:
    return json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))


def main():
    if len(sys.argv) < 2:
        print('Usage: python run_cli.py "<your task in plain english>"')
        print('Example: python run_cli.py "Open Notepad and type Hello World"')
        sys.exit(1)

    goal = " ".join(sys.argv[1:])
    config = load_config()

    print(f"⚡ PunkOS — CLI mode")
    print(f"Goal: {goal}\n")

    # Build provider (Ollama or Anthropic)
    try:
        provider = build_provider(config)
        print(f"✓ LLM provider: {provider.name()}")
    except Exception as e:
        print(f"❌ Failed to initialise LLM: {e}")
        sys.exit(1)

    # Wire up modules
    agent_cfg = config.get("agent", {})
    safety_cfg = config.get("safety", {})
    history_cfg = config.get("history", {})

    perception = VisualPerceptionEngine(
        max_elements=agent_cfg.get("max_elements", 40),
        max_dimension=agent_cfg.get("screenshot_max_dimension", 1280),
    )
    reasoning = ReasoningCore(provider)
    controller = DesktopController()
    # Build every optional subsystem (M6-M11). Anything unavailable comes back
    # as None with a warning rather than stopping the run.
    subs = build_subsystems(config, project_root=Path(__file__).parent, on_log=print)
    for warning in subs.warnings:
        print(f"⚠  {warning}")

    safety = SafetySystem(
        authorized_dirs=safety_cfg.get("authorized_directories", []),
        blocked_paths=safety_cfg.get("blocked_paths", []),
        confirm_medium_risk=safety_cfg.get("confirm_medium_risk", True),
        audit_log=subs.audit_log,
    )
    history_store = TaskHistoryStore(history_cfg.get("db_path", "data/history/punkos.db"))

    planner = StrategicPlanner(
        perception=perception,
        reasoning=reasoning,
        controller=controller,
        safety=safety,
        history_store=history_store,
        max_steps=agent_cfg.get("max_steps_per_task", 12),
        max_consecutive_failures=agent_cfg.get("max_consecutive_failures", 3),
        step_delay=agent_cfg.get("step_delay_seconds", 1.5),
        use_planning_phase=agent_cfg.get("use_planning_phase", True),
        **subs.as_planner_kwargs(),
    )

    result = planner.run(goal)

    print("\n" + "=" * 60)
    print(f"{'✅ SUCCESS' if result.success else '❌ FAILED'}")
    print(f"Task ID:  {result.task_id}")
    print(f"Mode:     {result.execution_mode}")
    print(f"Steps:    {result.steps_taken}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    print(f"Message:  {result.final_message}")
    if result.learning:
        print(f"Learned:  {result.learning}")
    print("=" * 60)

    # Clean up the browser if the web agent was used.
    if subs.web_agent is not None and subs.web_agent.is_running:
        subs.web_agent.close()


if __name__ == "__main__":
    main()

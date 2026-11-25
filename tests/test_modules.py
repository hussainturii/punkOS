"""
Unit tests for the modules added in the second half: M7 web agent, M8 plugins,
M9 sandbox + audit, M10 learning, M11 memory, and the sub-agent router.

Runs with no external dependencies — no Ollama, no Playwright, no Windows, no
network. Everything that would touch the outside world is either pure logic
(selector validation, field matching, routing) or exercised against a temp
directory (sqlite stores, sandbox subprocess).

Run:  python tests/test_modules.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Stub the Windows/GUI-only modules before importing core.
sys.modules.setdefault("mss", types.ModuleType("mss"))
_pag = types.ModuleType("pyautogui")
_pag.FAILSAFE = True
_pag.PAUSE = 0.05
for _name in ("click", "doubleClick", "write", "press", "hotkey", "scroll",
              "moveTo", "mouseDown", "mouseUp"):
    setattr(_pag, _name, lambda *a, **kw: None)
sys.modules.setdefault("pyautogui", _pag)

from core.audit import AuditLog                                   # noqa: E402
from core.embedding import TextEmbedder, cosine_similarity        # noqa: E402
from core.learning import ContinuousLearner, extract_steps        # noqa: E402
from core.memory import LearnedWorkflow, ProceduralMemory, WorkflowStep  # noqa: E402
from core.plugins import PluginRegistry, validate_manifest        # noqa: E402
from core.reasoning import NextAction                             # noqa: E402
from core.router import ExecutionMode, route                      # noqa: E402
from core.safety import SafetySystem                              # noqa: E402
from core.sandbox import SandboxExecutor, validate_script         # noqa: E402
from core.web_agent import FormField, WebAgent, validate_selector  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def assert_eq(actual, expected, label):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(value, label):
    if not value:
        raise AssertionError(f"{label}: expected truthy, got {value!r}")


def temp_db(name: str) -> str:
    path = Path(tempfile.gettempdir()) / f"punkos_test_{name}_{os.getpid()}.db"
    if path.exists():
        path.unlink()
    return str(path)


# ---------------------------------------------------------------------------
# M9 — safety classifier
# ---------------------------------------------------------------------------

def test_safety_no_false_positives():
    """The old classifier fired HIGH on 'dropdown' and 'payment'. It must not."""
    print("  test_safety_no_false_positives...")
    safety = SafetySystem([tempfile.gettempdir()], [])

    benign = NextAction(
        action_type="click", text="",
        reasoning="Click the dropdown to choose a preset payment option and save time",
    )
    assessment = safety.assess(benign)
    assert_eq(assessment.level.value, "LOW", "benign click risk level")
    assert_eq(assessment.requires_confirmation, False, "benign click confirmation")

    destructive = NextAction(action_type="type", text="delete", reasoning="remove it")
    assert_eq(safety.assess(destructive).level.value, "HIGH", "delete risk level")

    del_key = NextAction(action_type="key", text="shift+delete", reasoning="clear")
    assert_eq(safety.assess(del_key).level.value, "HIGH", "shift+delete risk level")

    flagged = NextAction(action_type="click", text="", reasoning="ok", is_high_risk=True)
    assert_eq(safety.assess(flagged).level.value, "HIGH", "LLM self-flag honoured")
    print("    ✓ pass")


def test_safety_boundary_enforcement():
    """SEC-2 / FE-3: blocklist wins, traversal is resolved before comparison."""
    print("  test_safety_boundary_enforcement...")
    root = Path(tempfile.gettempdir()) / "punkos_auth"
    root.mkdir(exist_ok=True)
    safety = SafetySystem([str(root)], [])

    ok, _ = safety.is_path_allowed(str(root / "notes.txt"))
    assert_true(ok, "file inside authorized dir is allowed")

    ok, reason = safety.is_path_allowed(r"C:\Windows\System32\bad.dll")
    assert_eq(ok, False, "system path blocked")
    assert_true("blocked" in reason.lower(), "blocked reason mentions blocking")

    ok, _ = safety.is_path_allowed(str(root / ".." / ".." / "Windows" / "x"))
    assert_eq(ok, False, "directory traversal rejected")

    # Sibling-prefix trap: /punkos_auth_evil must not count as inside /punkos_auth.
    ok, _ = safety.is_path_allowed(str(root) + "_evil/x.txt")
    assert_eq(ok, False, "sibling prefix directory rejected")

    all_ok, rejections = safety.authorize_paths([str(root / "a.txt"), r"C:\Windows\b"])
    assert_eq(all_ok, False, "batch authorization fails on one bad path")
    assert_eq(len(rejections), 1, "one rejection reported")
    print("    ✓ pass")


def test_safety_bulk_scope():
    """SDS Algorithm 5: >10 files is HIGH, 2-10 is MEDIUM."""
    print("  test_safety_bulk_scope...")
    root = Path(tempfile.gettempdir()) / "punkos_auth"
    root.mkdir(exist_ok=True)
    safety = SafetySystem([str(root)], [])
    action = NextAction(action_type="click", text="", reasoning="tidy up")

    many = [str(root / f"f{i}.txt") for i in range(15)]
    assert_eq(safety.assess(action, many).level.value, "HIGH", "15 files is HIGH")

    few = [str(root / f"f{i}.txt") for i in range(4)]
    assert_eq(safety.assess(action, few).level.value, "MEDIUM", "4 files is MEDIUM")

    one = [str(root / "f0.txt")]
    assert_eq(safety.assess(action, one).level.value, "LOW", "1 file is LOW")
    print("    ✓ pass")


# ---------------------------------------------------------------------------
# M9 — sandbox
# ---------------------------------------------------------------------------

def test_sandbox_rejects_dangerous_scripts():
    """SDS Algorithm 4 step 4: AST validation blocks every escape route."""
    print("  test_sandbox_rejects_dangerous_scripts...")
    auth = [tempfile.gettempdir()]

    dangerous = {
        "subprocess": "import subprocess\nsubprocess.run(['calc'])",
        "os.system": "import os\nos.system('dir')",
        "eval": "eval('1+1')",
        "exec": "exec('x=1')",
        "network": "import requests",
        "socket": "import socket",
        "dunder import": "__import__('os')",
        "escape write": "open('C:/Windows/x.dll', 'w')",
    }
    for label, script in dangerous.items():
        result = validate_script(script, auth)
        assert_eq(result.valid, False, f"{label} must be rejected")

    safe = "from pathlib import Path\nimport json\nprint(len(list(Path('.').glob('*'))))"
    assert_true(validate_script(safe, auth).valid, "safe script accepted")
    print("    ✓ pass")


def test_sandbox_executes_and_times_out():
    print("  test_sandbox_executes_and_times_out...")
    workdir = Path(tempfile.gettempdir()) / "punkos_sandbox_work"
    workdir.mkdir(exist_ok=True)
    executor = SandboxExecutor([str(workdir)], backend="subprocess", timeout_seconds=5)

    result = executor.execute("print('hello from sandbox')")
    assert_true(result.success, f"safe script should run: {result.error}")
    assert_true("hello from sandbox" in result.output, "stdout captured")
    assert_eq(result.exit_code, 0, "clean exit code")
    assert_true(len(result.script_sha256) == 64, "script hash recorded")

    blocked = executor.execute("import socket")
    assert_eq(blocked.success, False, "blocked script does not run")
    assert_true("validation" in blocked.error.lower(), "rejection explains why")

    timed_out = executor.execute("while True:\n    pass")
    assert_eq(timed_out.success, False, "infinite loop is terminated")
    assert_true("timeout" in timed_out.error.lower(), "timeout reported")
    print("    ✓ pass")


# ---------------------------------------------------------------------------
# M9 — audit log
# ---------------------------------------------------------------------------

def test_audit_chain_detects_tampering():
    """SEC-5: the hash chain must break visibly when a row is edited."""
    print("  test_audit_chain_detects_tampering...")
    path = temp_db("audit")
    log = AuditLog(path)

    log.record_confirmation("click", "HIGH", True, "deletes files")
    log.record_confirmation("type", "MEDIUM", False, "overwrites a file")
    log.record_sandbox("a" * 64, 0, 1.25)
    assert_eq(log.entry_count(), 3, "three entries recorded")

    intact, problem = log.verify_chain()
    assert_true(intact, f"fresh chain should verify: {problem}")

    conn = sqlite3.connect(path)
    conn.execute("UPDATE audit SET detail = 'nothing happened' WHERE seq = 2")
    conn.commit()
    conn.close()

    intact, problem = log.verify_chain()
    assert_eq(intact, False, "tampering must be detected")
    assert_true("2" in problem, "the tampered row is identified")

    cancelled = log.list_entries(event="high_risk_cancelled")
    assert_eq(len(cancelled), 1, "cancelled actions are queryable")
    os.remove(path)
    print("    ✓ pass")


# ---------------------------------------------------------------------------
# M7 — web agent (pure logic; no browser needed)
# ---------------------------------------------------------------------------

def test_web_selector_validation():
    """SEC-8: reject injection-shaped selectors, accept ordinary CSS."""
    print("  test_web_selector_validation...")
    for good in ("#email", "input[name='user']", "div.card > a:nth-of-type(2)",
                 "text=Submit", ".a .b"):
        ok, reason = validate_selector(good)
        assert_true(ok, f"{good!r} should be valid ({reason})")

    for bad in ("javascript:alert(1)", "<script>x</script>", "a; eval(1)",
                "div onload=x", "", "x" * 600, "document.cookie"):
        ok, _ = validate_selector(bad)
        assert_eq(ok, False, f"{bad[:30]!r} should be rejected")
    print("    ✓ pass")


def test_web_field_matching():
    """FR5.6: match fields whose ids are complex, missing or generated."""
    print("  test_web_field_matching...")
    values = {
        "email": "ada@example.com",
        "password": "pw1",
        "confirm password": "pw2",
        "full name": "Ada Lovelace",
    }
    cases = [
        (FormField("#1", "Email Address", "email"), "ada@example.com"),
        (FormField("#2", "", "password", placeholder="Enter your password"), "pw1"),
        (FormField("#3", "Confirm Password", "password"), "pw2"),
        (FormField("#4", "", "text", name="full_name_x9f2"), "Ada Lovelace"),
        (FormField("#5", "", "text", name="fullName"), "Ada Lovelace"),
        (FormField("#6", "", "text", name="username"), None),
        (FormField("#7", "Country", "select"), None),
    ]
    for field, expected in cases:
        got = WebAgent.match_field(field, values)
        label = field.label or field.placeholder or field.name
        assert_eq(got, expected, f"field {label!r}")
    print("    ✓ pass")


def test_web_agent_degrades_without_playwright():
    """A missing optional dependency must not crash PunkOS."""
    print("  test_web_agent_degrades_without_playwright...")
    agent = WebAgent()
    assert_eq(agent.is_running, False, "agent starts stopped")
    result = agent.navigate("example.com")
    assert_eq(result.success, False, "navigate without start fails cleanly")
    assert_eq(agent.screenshot_b64(), "", "screenshot returns empty, not an exception")
    assert_eq(agent.dom_snapshot(), None, "dom snapshot returns None, not an exception")
    print("    ✓ pass")


def test_web_url_normalization():
    """A bare hostname gets https://; every other scheme is left alone.

    Regression: blindly prefixing https:// turned file:///C:/page.html into
    https://file:///C:/page.html, so local pages could not be opened.
    """
    print("  test_web_url_normalization...")
    cases = [
        ("example.com", "https://example.com"),
        ("www.example.com/x", "https://www.example.com/x"),
        ("//example.com", "https://example.com"),
        ("http://example.com", "http://example.com"),
        ("https://example.com", "https://example.com"),
        ("file:///C:/Users/x/page.html", "file:///C:/Users/x/page.html"),
        ("data:text/html,<p>hi</p>", "data:text/html,<p>hi</p>"),
        ("about:blank", "about:blank"),
        ("", ""),
    ]
    for raw, expected in cases:
        assert_eq(WebAgent.normalize_url(raw), expected, f"normalize {raw!r}")
    print("    ✓ pass")


def test_web_scrape_export_formats():
    print("  test_web_scrape_export_formats...")
    records = [{"title": "Widget", "price": "$9"}, {"title": "Gadget", "price": "$12"}]
    csv_out = WebAgent.to_csv(records)
    assert_true("title,price" in csv_out, "csv header present")
    assert_true("Widget" in csv_out, "csv row present")
    json_out = WebAgent.to_json(records)
    assert_true('"title"' in json_out, "json keys present")
    assert_eq(WebAgent.to_csv([]), "", "empty scrape yields empty csv")
    print("    ✓ pass")


# ---------------------------------------------------------------------------
# M8 — plugins
# ---------------------------------------------------------------------------

def test_plugin_manifest_validation():
    """SEC-9: a malicious or malformed manifest is rejected with reasons."""
    print("  test_plugin_manifest_validation...")
    bad = {
        "plugin_id": "Evil Plugin!!",
        "app_name": "X",
        "version": "not-a-version",
        "description": "d",
        "knowledge_base": ["../../../etc/passwd"],
        "workflows": [{"name": "w", "trigger_patterns": [],
                       "steps": [{"action_type": "exec_shell", "text": "rm -rf /"}]}],
    }
    issues = validate_manifest(bad)
    joined = " ".join(str(i) for i in issues)
    assert_true(len(issues) >= 4, f"expected several issues, got {len(issues)}")
    assert_true("plugin_id" in joined, "bad id reported")
    assert_true("version" in joined, "bad version reported")
    assert_true("exec_shell" in joined, "illegal action reported")
    assert_true("knowledge_base" in joined, "path traversal reported")

    assert_true(validate_manifest("not a dict"), "non-dict manifest rejected")
    print("    ✓ pass")


def test_shipped_plugins_are_valid():
    """Every plugin we ship must pass its own validator and dry-run test."""
    print("  test_shipped_plugins_are_valid...")
    plugin_dir = Path(__file__).parent.parent / "data" / "plugins"
    registry = PluginRegistry(plugin_dir)
    count = registry.discover()

    assert_true(count >= 3, f"expected at least 3 shipped plugins, found {count}")
    assert_eq(registry.rejected, {}, "no shipped plugin should fail validation")

    for plugin_id in registry.plugins:
        ok, report = registry.test_plugin(plugin_id)
        assert_true(ok, f"plugin {plugin_id} failed its dry run: {report}")
    print("    ✓ pass")


def test_plugin_routing_and_stats():
    print("  test_plugin_routing_and_stats...")
    plugin_dir = Path(__file__).parent.parent / "data" / "plugins"
    registry = PluginRegistry(plugin_dir)
    registry.discover()

    match = registry.find_workflow("open file utils.py", "main.py - Visual Studio Code")
    assert_true(match is not None, "VS Code workflow should be found")
    plugin, workflow = match
    assert_eq(plugin.plugin_id, "vscode", "correct plugin selected")
    assert_eq(workflow.name, "open_file_by_name", "correct workflow selected")

    hint = registry.hint_for_prompt("main.py - Visual Studio Code", "open file utils.py")
    assert_true("APP EXPERTISE" in hint, "hint has a header")
    assert_true("ctrl+p" in hint, "hint contains the real keystroke")
    assert_true("file_name" in hint, "hint marks the runtime parameter")

    assert_eq(registry.hint_for_prompt("Calculator", "add numbers"), "",
              "no hint when no plugin matches")

    # An unmatched window must not borrow another app's plugin.
    assert_eq(registry.plugin_for_window("Calculator"), None, "no plugin for Calculator")
    print("    ✓ pass")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def test_router_modes():
    """M3 FE-2 / M4 FE-3: goals reach the right sub-agent."""
    print("  test_router_modes...")
    cases = [
        ("Open Calculator and compute 2+2", ExecutionMode.DESKTOP),
        ("Open Notepad and type Hello World", ExecutionMode.DESKTOP),
        ("Go to github.com and search for playwright", ExecutionMode.WEB),
        ("Scrape product titles from example.com", ExecutionMode.WEB),
        ("Fill out the form on the contact page", ExecutionMode.WEB),
        ("Rename all 250 files in Downloads", ExecutionMode.CODE),
        ("Organise my Downloads folder by file type", ExecutionMode.CODE),
        # The trap: 'delete all the text' is editing, not a bulk file op.
        ("Delete all the text in this document", ExecutionMode.DESKTOP),
    ]
    for goal, expected in cases:
        decision = route(goal)
        assert_eq(decision.mode, expected, f"routing {goal!r}")
        assert_true(decision.reason, f"routing {goal!r} must explain itself")
    print("    ✓ pass")


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

@patch.object(TextEmbedder, "_load", new=lambda self: None)
def test_embedding_fallback():
    """The fallback must be unit-length, deterministic, and self-consistent."""
    print("  test_embedding_fallback...")
    embedder = TextEmbedder()
    a = embedder.encode("open the calculator")
    b = embedder.encode("open the calculator")
    c = embedder.encode("send an email to my manager")

    assert_eq(len(a), 384, "vector dimension")
    assert_true(abs(sum(x * x for x in a) - 1.0) < 1e-5, "vector is unit length")
    assert_true(abs(cosine_similarity(a, b) - 1.0) < 1e-6, "identical text scores 1.0")
    assert_true(cosine_similarity(a, c) < 0.5, "unrelated text scores low")
    assert_true(embedder.backend_name(), "backend is named")
    print("    ✓ pass")


# ---------------------------------------------------------------------------
# M11 — procedural memory
# ---------------------------------------------------------------------------

@patch.object(TextEmbedder, "_load", new=lambda self: None)
def test_memory_persistence_and_stats():
    print("  test_memory_persistence_and_stats...")
    path = temp_db("memory")
    memory = ProceduralMemory(path)

    workflow = LearnedWorkflow(
        workflow_id="wf1", name="Open Notepad", trigger_pattern="open notepad",
        steps=[WorkflowStep("key", "win+r"), WorkflowStep("type", "notepad")],
        execution_count=1, success_count=1,
    )
    memory.save_workflow(workflow)

    memory.set_preference("theme", "dark")
    memory.note_app_usage("Notepad")
    memory.note_app_usage("Notepad")
    memory.add_correction("task1", 0, "type notepad", "type notepad.exe", "full name")
    memory.set_bookmark("task1", True)

    # Reopen from scratch — everything must still be there.
    reopened = ProceduralMemory(path)
    assert_eq(len(reopened.list_workflows()), 1, "workflow persisted")
    assert_eq(reopened.get_preference("theme"), "dark", "preference persisted")
    assert_eq(reopened.correction_count(), 1, "correction persisted")
    assert_true("task1" in reopened.list_bookmarks(), "bookmark persisted")
    assert_eq(reopened.frequent_apps()[0], ("Notepad", 2), "app usage counted")

    stats = reopened.statistics()
    assert_eq(stats["learned_workflows"], 1, "stats: workflows")
    assert_eq(stats["corrections_applied"], 1, "stats: corrections")
    assert_true(stats["store_size_bytes"] > 0, "stats: store size")

    reopened.set_bookmark("task1", False)
    assert_eq(reopened.is_bookmarked("task1"), False, "bookmark removable")
    os.remove(path)
    print("    ✓ pass")


# ---------------------------------------------------------------------------
# M10 — continuous learning
# ---------------------------------------------------------------------------

def _history(*pairs):
    return [{"action": action, "success": success} for action, success in pairs]


def test_learning_extracts_only_successful_steps():
    print("  test_learning_extracts_only_successful_steps...")
    history = _history(
        ("key tag=None text='win+r'", True),
        ("type tag=3 text='notepad'", True),
        ("click tag=9 text=''", False),        # failed — must be dropped
        ("wait tag=None text=''", True),       # non-procedural — must be dropped
    )
    steps = extract_steps(history)
    assert_eq(len(steps), 2, "only successful, procedural steps kept")
    assert_eq(steps[0].action_type, "key", "first step action")
    assert_eq(steps[0].text, "win+r", "repr quotes stripped from text")
    assert_eq(steps[1].text, "notepad", "second step text")
    print("    ✓ pass")


@patch.object(TextEmbedder, "_load", new=lambda self: None)
def test_learning_create_merge_and_correct():
    """SDS Algorithm 6 end to end."""
    print("  test_learning_create_merge_and_correct...")
    path = temp_db("learning")
    memory = ProceduralMemory(path)
    learner = ContinuousLearner(memory)

    history = _history(
        ("key tag=None text='win+r'", True),
        ("type tag=3 text='notepad'", True),
        ("key tag=None text='enter'", True),
        ("completion_check", True),  # Synthetic verification required by learning.
    )

    first = learner.learn_from_task("t1", "Open Notepad", history, success=True)
    assert_eq(first.action, "created", "first run creates a workflow")

    second = learner.learn_from_task("t2", "Open Notepad", history, success=True)
    assert_eq(second.action, "merged", "identical goal merges")
    assert_eq(second.workflow_id, first.workflow_id, "merged into the same workflow")

    workflow = memory.get_workflow(first.workflow_id)
    assert_eq(workflow.execution_count, 2, "execution count incremented")
    assert_eq(workflow.success_rate, 1.0, "success rate recomputed")

    # A failed run teaches nothing.
    failed = learner.learn_from_task("t3", "Something else entirely", history, success=False)
    assert_eq(failed.learned, False, "failed task is not learned")

    # A user correction replaces the step it fixed and is flagged.
    correction = memory.add_correction("t4", 1, "type notepad", "type notepad.exe")
    learner.learn_from_task("t4", "Open Notepad", history, success=True,
                            corrections=[correction])
    workflow = memory.get_workflow(first.workflow_id)
    corrected = [s for s in workflow.steps if s.from_correction]
    assert_eq(len(corrected), 1, "exactly one correction-derived step")
    assert_eq(corrected[0].text, "notepad.exe", "correction applied")

    # Forced save (FR7.3 'Save as Template') works even on a short/failed run.
    forced = learner.learn_from_task("t5", "Tiny task",
                                     _history(("key tag=None text='f5'", True)),
                                     success=False, force=True)
    assert_true(forced.learned, "forced save stores regardless of outcome")

    os.remove(path)
    print("    ✓ pass")


@patch.object(TextEmbedder, "_load", new=lambda self: None)
def test_learning_recall_into_prompt():
    print("  test_learning_recall_into_prompt...")
    path = temp_db("recall")
    memory = ProceduralMemory(path)
    learner = ContinuousLearner(memory)

    history = _history(
        ("key tag=None text='win+r'", True),
        ("type tag=3 text='calc'", True),
        ("completion_check", True),  # Input receipts alone must not teach a workflow.
    )
    learner.learn_from_task("t1", "Open Calculator", history, success=True)

    context = memory.context_for_prompt("Open Calculator")
    assert_true("REMEMBERED PROCEDURE" in context, "recall header present")
    assert_true("win+r" in context, "recalled steps present")

    assert_eq(memory.context_for_prompt("completely unrelated aardvark topic"), "",
              "no recall for an unrelated goal")
    os.remove(path)
    print("    ✓ pass")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    print("─" * 60)
    print("PunkOS Module Tests (M7-M11 + router)")
    print("─" * 60)
    tests = [
        test_safety_no_false_positives,
        test_safety_boundary_enforcement,
        test_safety_bulk_scope,
        test_sandbox_rejects_dangerous_scripts,
        test_sandbox_executes_and_times_out,
        test_audit_chain_detects_tampering,
        test_web_selector_validation,
        test_web_field_matching,
        test_web_agent_degrades_without_playwright,
        test_web_url_normalization,
        test_web_scrape_export_formats,
        test_plugin_manifest_validation,
        test_shipped_plugins_are_valid,
        test_plugin_routing_and_stats,
        test_router_modes,
        test_embedding_fallback,
        test_memory_persistence_and_stats,
        test_learning_extracts_only_successful_steps,
        test_learning_create_merge_and_correct,
        test_learning_recall_into_prompt,
    ]
    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"    ✗ FAIL: {e}")
            failed.append((test.__name__, e))

    print()
    print("─" * 60)
    if failed:
        print(f"❌ {len(failed)}/{len(tests)} tests failed:")
        for name, e in failed:
            print(f"   - {name}: {e}")
        sys.exit(1)
    print(f"✅ All {len(tests)} tests passed.")
    print("─" * 60)


if __name__ == "__main__":
    main()

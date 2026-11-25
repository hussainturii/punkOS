"""
M4: Strategic Planning System
The agent loop that ties everything together.

Week 1 features:
- Perceive → Reason → Safety → Execute loop
- Per-step decisions
- Console confirmation for HIGH/MEDIUM risk
- Step limit + consecutive failure detection

Week 2 features:
- Upfront task plan: ask LLM for a 3-8 step plan before execution starts
- Plan-aware execution: each step decision sees the full plan + which step is current
- Self-healing replan: when stuck, ask the LLM for a different approach
- Task history persistence to SQLite
- Pluggable callbacks for UI integration (logs, status, confirmations)
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from .perception import VisualPerceptionEngine
from .reasoning import ReasoningCore, NextAction, TaskPlan
from .controller import DesktopController, ExecutionResult
from .safety import SafetySystem, RiskAssessment, RiskLevel, cli_confirm
from .history import TaskHistoryStore, TaskRecord
from .router import ExecutionMode, RoutingDecision, route
from .agent_checks import AttemptGuard, observation, verify_completion, mouse_only, redundant_calculator_input, misplaced_save_action, desktop_attempt_signature, misplaced_document_edit, protected_source_save, open_file_dialog, unfinished_replacement_save
from .agent_checks import requested_editor_replacement


@dataclass
class TaskResult:
    task_id: str
    success: bool
    final_message: str
    steps_taken: int
    plan: Optional[TaskPlan]
    history: list[dict] = field(default_factory=list)
    duration_seconds: float = 0.0
    execution_mode: str = "DESKTOP"
    routing_reason: str = ""
    learning: Optional[str] = None
    metrics: dict = field(default_factory=dict)


# Type aliases for callbacks (used by the UI in week 3)
LogCallback = Callable[[str], None]
StatusCallback = Callable[[str], None]
ConfirmCallback = Callable[[NextAction, RiskAssessment], bool]
PlanCallback = Callable[[TaskPlan], None]
StepCallback = Callable[[int, NextAction, ExecutionResult], None]
PerceiveCallback = Callable[[object, Optional[int]], None]   # (ScreenState, target_tag)


class StrategicPlanner:
    """The main agent loop with upfront planning and self-healing."""

    def __init__(
        self,
        perception: VisualPerceptionEngine,
        reasoning: ReasoningCore,
        controller: DesktopController,
        safety: SafetySystem,
        history_store: Optional[TaskHistoryStore] = None,
        max_steps: int = 15,
        max_consecutive_failures: int = 3,
        step_delay: float = 0.6,
        use_planning_phase: bool = True,
        use_memory_recall: bool = False,
        # Optional subsystems. All default to None so the planner still runs
        # standalone — the integration tests construct it without any of them.
        memory=None,              # M11 ProceduralMemory
        learner=None,             # M10 ContinuousLearner
        plugins=None,             # M8  PluginRegistry
        sandbox=None,             # M9  SandboxExecutor
        web_agent=None,           # M7  WebAgent
        audit_log=None,           # M9  AuditLog
        on_log: Optional[LogCallback] = None,
        on_status: Optional[StatusCallback] = None,
        on_confirm: Optional[ConfirmCallback] = None,
        on_plan_ready: Optional[PlanCallback] = None,
        on_step_complete: Optional[StepCallback] = None,
        on_perceive: Optional[PerceiveCallback] = None,
        desktop_handoff=None,
        planning_mode='always',
        max_task_seconds=600,
        file_search=None,
    ):
        self.perception = perception
        self.reasoning = reasoning
        self.controller = controller
        self.safety = safety
        self.history_store = history_store
        self.max_steps = max_steps
        self.max_consecutive_failures = max_consecutive_failures
        self.step_delay = step_delay
        self.use_planning_phase = use_planning_phase
        self.use_memory_recall = use_memory_recall
        self.memory = memory
        self.learner = learner
        self.plugins = plugins
        self.sandbox = sandbox
        self.web_agent = web_agent
        self.file_search = file_search
        self.audit_log = audit_log
        self.desktop_handoff = desktop_handoff
        self.planning_mode = planning_mode
        self.max_task_seconds = max(1, float(max_task_seconds))
        self._deadline = None
        self._stop_message = 'User stopped task'

        self.on_log = on_log or print
        self.on_status = on_status or (lambda _s: None)
        self.on_confirm = on_confirm or cli_confirm
        self.on_plan_ready = on_plan_ready or (lambda _p: None)
        self.on_step_complete = on_step_complete or (lambda _i, _a, _r: None)
        self.on_perceive = on_perceive or (lambda _state, _target: None)

        self._stop_requested = False

    def request_stop(self):
        """Called by the UI's emergency stop button."""
        self._stop_requested = True

    def is_stopped(self):
        if not self._stop_requested and self._deadline is not None and time.monotonic() >= self._deadline:
            self._stop_requested = True
            self._stop_message = 'Task time limit exceeded'
        return self._stop_requested

    def _set_status(self, status: str):
        self.on_status(status)
        self.on_log(f"[{status}]")

    def run(self, goal: str) -> TaskResult:
        from .llm import OllamaProvider
        from .cloud_llm import CloudJSONProvider
        provider = self.reasoning.provider
        self._deadline = time.monotonic() + self.max_task_seconds
        previous_cancel = getattr(provider, 'cancel_check', None)
        previous_retry = getattr(provider, 'on_retry', None)
        if isinstance(provider, CloudJSONProvider):
            provider.on_retry = self.on_log
        if isinstance(provider, (OllamaProvider, CloudJSONProvider)):
            provider.cancel_check = self.is_stopped
        try:
            from contextlib import ExitStack
            from .metrics import TaskMetrics
            self._metrics = TaskMetrics()
            with ExitStack() as stack:
                for obj, method, category in ((provider, 'call_with_schema', 'model'),
                        (self.perception, 'perceive', 'perception'),
                        (self.controller, 'execute', 'desktop_action'),
                        (self.web_agent, 'task_observation', 'web_observation'),
                        (self.web_agent, 'task_action', 'web_action')):
                    stack.enter_context(self._metrics.instrument(obj, method, category))
                result = self._run(goal)
            result.metrics = self._metrics.snapshot()
            model = result.metrics['categories'].get('model', {})
            self.on_log(f"[metrics] Model calls: {model.get('calls', 0)}; model time: {model.get('seconds', 0):.1f}s; task time: {result.metrics['elapsed_seconds']:.1f}s")
            from .task_report import partial_report
            result.final_message = partial_report(result.final_message, result.success, result.plan, result.history)
            return result
        finally:
            if isinstance(provider, CloudJSONProvider):
                provider.on_retry = previous_retry
            if isinstance(provider, (OllamaProvider, CloudJSONProvider)):
                provider.cancel_check = previous_cancel if previous_cancel is not None else (
                    (lambda: False) if isinstance(provider, OllamaProvider) else None)

    def _run(self, goal: str) -> TaskResult:
        """Execute the agent loop end-to-end."""
        task_id = uuid.uuid4().hex[:12]
        start_time = time.monotonic()
        if self.is_stopped():
            self._record_history(task_id, goal, None, [], False, self._stop_message, 0, 0)
            return TaskResult(task_id, False, self._stop_message, 0, None)

        self.on_log("\n" + "=" * 60)
        self.on_log(f"🎯 GOAL: {goal}")
        self.on_log(f"   Task ID: {task_id}")
        self.on_log(f"   Provider: {self.reasoning.provider.name()}")
        self.on_log("=" * 60)

        from .file_destination import existing_file_stop
        preflight = existing_file_stop(goal)
        if preflight is not None:
            success, message = preflight
            history = [{'action': 'destination_preflight', 'success': success, 'outcome': message}]
            duration = time.monotonic() - start_time
            self._record_history(task_id, goal, None, history, success, message, 0, duration)
            self._set_status('Completed' if success else 'Failed')
            self.on_log(message)
            return TaskResult(task_id, success, message, 0, None, history, duration)

        from .llm import OllamaProvider
        if isinstance(self.reasoning.provider, OllamaProvider):
            self._set_status('Checking local model')
            try:
                self.reasoning.provider.ensure_available()
            except Exception as exc:
                message = self._stop_message if self.is_stopped() else f'Local model unavailable: {exc}. Start Ollama and check the configured model. No application input was issued.'
                duration = time.monotonic() - start_time
                self._record_history(task_id, goal, None, [], False, message, 0, duration)
                self._set_status('Stopped' if self.is_stopped() else 'Failed')
                return TaskResult(task_id, False, message, 0, None, [], duration)

        # ───────────── Phase 0: Route to a sub-agent (M3 FE-2 / M4 FE-3) ─────────────
        from .interaction_policy import InteractionPolicy
        interaction = InteractionPolicy(goal)
        target_app = ''
        routing = route(goal)
        from .screen_inspection import inspection_only, inspect_screen
        browser_handoff = None
        from .window_handoff import DesktopHandoff
        if routing.mode is ExecutionMode.WEB and isinstance(self.desktop_handoff, DesktopHandoff):
            browser_handoff = self.desktop_handoff.reuse_browser(goal)
            if browser_handoff is not None:
                routing = RoutingDecision(ExecutionMode.DESKTOP, 1.0,
                    'An existing browser session takes priority; using desktop interaction instead of a guest browser.')
        elif routing.mode is ExecutionMode.DESKTOP and isinstance(self.desktop_handoff, DesktopHandoff):
            from .window_handoff import requested_apps
            if not requested_apps(goal):
                browser_handoff = self.desktop_handoff.reuse_browser(goal, match_open_title=True)
                if browser_handoff is not None:
                    routing = RoutingDecision(ExecutionMode.DESKTOP, .95,
                        'The referenced open page matches an observed browser window; using its existing session.')
        if inspection_only(goal) and routing.mode is ExecutionMode.WEB:
            routing = RoutingDecision(ExecutionMode.DESKTOP, .95,
                'Observation-only request: inspect the current desktop instead of starting a web action loop.')
        self.on_log(f"\n🧭 Execution mode: {routing.mode.value} ({routing.confidence:.0%})")
        self.on_log(f"   {routing.reason}")

        if routing.mode is ExecutionMode.FILE_OPERATION:
            from .file_operations import run_file_operation
            from .file_batch import needs_batch, run_batch
            self._set_status('Working with authorized files')
            if needs_batch(goal):
                def review_files(preview, operations):
                    self._set_status('Review file plan')
                    action = NextAction('file_plan', text=preview)
                    assessment = RiskAssessment(RiskLevel.HIGH if len(operations) > 10 else RiskLevel.MEDIUM,
                        'Review the exact source and destination paths before changing files.', True,
                        preview=[('modify', line) for line in preview.splitlines()],
                        affected_paths=[str(dest) for _,_,dest in operations])
                    return self.on_confirm(action, assessment)
                success, message, history = run_batch(goal, self.reasoning.provider, self.safety,
                    self.is_stopped, self.on_log, review_files)
            else:
                success, message, history = run_file_operation(goal, self.reasoning.provider,
                                                               self.safety, self.is_stopped, self.on_log)
            duration = time.monotonic() - start_time
            self._record_history(task_id, goal, None, history, success, message, len(history), duration, allow_learning=False)
            self._set_status('Completed' if success else 'Stopped' if self.is_stopped() else 'Needs attention')
            self.on_log(message)
            return TaskResult(task_id, success, message, len(history), None, history, duration,
                              execution_mode='FILE_OPERATION', routing_reason=routing.reason)

        if routing.mode is ExecutionMode.FILE_SEARCH:
            from .file_tasks import run_file_task
            self._set_status('Searching authorized files')
            success, message, history = run_file_task(goal, self.reasoning.provider, self.file_search,
                                                      self.safety, self.is_stopped, self.on_log)
            duration = time.monotonic() - start_time
            self._record_history(task_id, goal, None, history, success, message, len(history), duration, allow_learning=False)
            self._set_status('Completed' if success else 'Stopped' if self.is_stopped() else 'Needs attention')
            self.on_log(message)
            return TaskResult(task_id, success, message, len(history), None, history, duration,
                              execution_mode='FILE_SEARCH', routing_reason=routing.reason)

        if routing.mode is ExecutionMode.WEB and self.web_agent is not None:
            return self._run_web_task(task_id, goal, routing, start_time)
        if routing.mode is ExecutionMode.CODE and self.sandbox is not None:
            return self._run_code_task(task_id, goal, routing, start_time)
        if routing.mode is not ExecutionMode.DESKTOP:
            self.on_log(f"   ⚠ {routing.mode.value} sub-agent not available — "
                        f"falling back to desktop control.")
            routing = RoutingDecision(ExecutionMode.DESKTOP, 0.3,
                                      f"{routing.mode.value} agent unavailable; using desktop")

        if self.desktop_handoff is not None:
            self._set_status('Focusing target')
            handoff = browser_handoff if browser_handoff is not None else self.desktop_handoff(goal)
            target_app = getattr(handoff, 'target_app', '')
            interaction.launch_complete = getattr(handoff, 'target_verified', False) is True
            self.on_log(handoff.message)
            if not handoff.success or self.is_stopped():
                message = self._stop_message if self.is_stopped() else handoff.message
                duration = time.monotonic() - start_time
                self._record_history(task_id, goal, None, [], False, message, 0, duration)
                self._set_status('Failed')
                return TaskResult(task_id, False, message, 0, None, [], duration)
            if handoff.complete:
                duration = time.monotonic() - start_time
                history = [{'action': 'open_app', 'outcome': handoff.message, 'success': True}]
                self._record_history(task_id, goal, None, history, True, handoff.message, 1, duration)
                self._set_status('Completed')
                return TaskResult(task_id, True, handoff.message, 1, None, history, duration)

        from .document_evidence import DocumentEvidence
        document_evidence = DocumentEvidence(goal)

        # ───────────── Phase 1: Initial perception ─────────────
        self._set_status("Perceiving")
        self.on_log("\n👁  Capturing initial screen state...")
        initial_state = self.perception.perceive()
        self.on_log(f"   Window: {initial_state.foreground_window_title}")
        self.on_log(f"   Elements: {len(initial_state.elements)}")
        if inspection_only(goal):
            self._set_status('Inspecting current screen')
            success, message, inspection_history = inspect_screen(
                goal, initial_state, self.reasoning.provider, self.is_stopped)
            duration = time.monotonic() - start_time
            self._record_history(task_id, goal, None, inspection_history, success, message,
                                 0, duration, allow_learning=False)
            self._set_status('Completed' if success else 'Stopped' if self.is_stopped() else 'Needs attention')
            self.on_log(message)
            return TaskResult(task_id, success, message, 0, None, inspection_history, duration)

        # M11: remember which apps this user actually works in.
        if self.memory is not None and initial_state.foreground_window_title:
            try:
                self.memory.note_app_usage(initial_state.foreground_window_title)
            except Exception as e:
                self.on_log(f"[memory] could not record app usage: {e}")

        # M11 FE-1: retrieve a remembered procedure for this goal, if any.
        memory_context = ""
        if self.memory is not None and self.use_memory_recall:
            try:
                memory_context = self.memory.context_for_prompt(goal)
                if memory_context:
                    self.on_log("🧠 Recalled a matching procedure from memory.")
            except Exception as e:
                self.on_log(f"[memory] recall failed: {e}")

        # ───────────── Phase 2: Upfront planning (Week 2) ─────────────
        plan: Optional[TaskPlan] = None
        from .task_policy import needs_plan
        if self.use_planning_phase and (self.planning_mode != 'auto' or needs_plan(goal)):
            self._set_status("Planning")
            self.on_log("\n🧭 Asking LLM for a multi-step plan...")
            plan = self.reasoning.plan_task(goal, initial_state)
            if self.is_stopped():
                duration = time.monotonic() - start_time
                self._record_history(task_id, goal, None, [], False, self._stop_message, 0, duration)
                return TaskResult(task_id, False, self._stop_message, 0, None, [], duration)
            if plan is None:
                self.on_log("   ⚠ Planning failed — falling back to step-by-step mode.")
            elif not plan.feasible:
                self.on_log(f"   ✋ LLM declined task: {plan.summary}")
                duration = time.monotonic() - start_time
                self._record_history(task_id, goal, plan, [], False, plan.summary, 0, duration)
                return TaskResult(task_id, False, f"LLM declined: {plan.summary}", 0, plan, [], duration)
            else:
                self.on_log(f"   📋 Plan: {plan.summary}")
                for s in plan.steps:
                    risk_marker = " ⚠HIGH" if s.is_high_risk else ""
                    self.on_log(f"      {s.step_number}. {s.description}{risk_marker}")
                self.on_plan_ready(plan)

        # ───────────── Phase 3: Step-by-step execution loop ─────────────
        history: list[dict] = []
        consecutive_failures = 0
        plan_step_idx = 0
        attempts = AttemptGuard()
        from .task_ledger import TaskLedger
        ledger = TaskLedger()
        from .input_evidence import InputEvidence
        input_evidence = InputEvidence()
        from .requirements import requirement_context
        requirements = plan.requirements if plan else []
        assessments = {}
        rejected_completions = 0
        transport_failures = 0

        for step in range(1, self.max_steps + 1):
            if self.is_stopped():
                self._set_status("Stopped")
                self.on_log("\n🛑 Stop requested — halting.")
                duration = time.monotonic() - start_time
                self._record_history(task_id, goal, plan, history, False, self._stop_message, step - 1, duration)
                return TaskResult(task_id, False, self._stop_message, step - 1, plan, history, duration)

            self.on_log(f"\n--- Step {step}/{self.max_steps} ---")

            # PERCEIVE (skip on first iteration — we already have initial_state)
            self._set_status(f"Perceiving (step {step})")
            state = initial_state if step == 1 else self.perception.perceive()
            self.on_log(f"👁  Window: {state.foreground_window_title} | {len(state.elements)} elements")
            diagnostic = getattr(state, 'accessibility_diagnostic', '')
            if isinstance(diagnostic, str) and diagnostic:
                self.on_log(diagnostic)
            self.on_perceive(state, None)
            if interaction.scope == 'launch' and not interaction.launch_complete and target_app:
                from .window_handoff import APPS
                current = self.controller._windows().foreground()
                if (current and current.handle == state.foreground_window_handle and
                        current.executable.casefold() in APPS.get(target_app, set())):
                    interaction.launch_complete = True
            output_recovery = document_evidence.allow_output_recovery(state, history)
            source_prerequisite = document_evidence.source_prerequisite(state, history)
            document_evidence.observe_progress(state, history)
            previous_input_detail = input_evidence.detail
            input_receipt = input_evidence.observe(state, self.reasoning.provider, self.is_stopped)
            if not input_receipt and input_evidence.detail != previous_input_detail:
                self.on_log('[input verification] ' + input_evidence.detail)
            if input_receipt:
                history.append({'action': 'input_readback', 'success': True, 'outcome': input_receipt})
                self.on_log(input_receipt)
            progress_hint = document_evidence.progress_hint()
            progress_hint += '\n' + input_evidence.hint()
            ledger.observe(observation(state), history)
            progress_hint += '\n' + ledger.context()
            progress_hint += requirement_context(requirements, assessments)
            if browser_handoff is not None and browser_handoff.success:
                progress_hint += ('\nBROWSER SESSION: The existing user browser was focused for this task. '
                                  'Work in that session using observed desktop controls; do not launch '
                                  'a new browser, guest/incognito session, or switch profiles. '
                                  'Use normal navigation shortcuts only when the interaction policy permits. '
                                  'If required page controls are unobservable, report the limitation instead '
                                  'of guessing coordinates or claiming success. Do not close the user browser.')

            # REASON
            self._set_status(f"Thinking (step {step})")
            self.on_log("🧠 Asking LLM for next action...")
            try:
                if consecutive_failures >= 2:
                    last_outcome = history[-1]["outcome"] if history else "unknown"
                    if self.plugins is not None:
                        try:
                            last_outcome += '\n' + self.plugins.hint_for_prompt(state.foreground_window_title, goal)
                        except Exception as exc:
                            self.on_log(f'[plugins] recovery hint lookup failed: {exc}')
                    action = self.reasoning.replan_after_failure(goal, state, history, last_outcome + '\n' + progress_hint + '\nINTERACTION POLICY: ' + interaction.hint())
                    self.on_log("   (recovery mode)")
                else:
                    # M8: if a plugin owns this window, hand the model its
                    # expert workflow for the current goal.
                    plugin_hint = ""
                    if source_prerequisite:
                        plugin_hint = 'DOCUMENT PREREQUISITE: ' + source_prerequisite
                    elif output_recovery:
                        plugin_hint = ('OUTPUT REVIEW: This task already reopened the exact requested output '
                                       'and matched its editor contents against disk at that point. '
                                       'Compare the current editor with the goal. If all requested changes '
                                       'are present and no later edits were made, return DONE with the '
                                       'requested report now; do not save or reopen again. Only if something '
                                       'is still missing, correct it in this output tab and save those new '
                                       'edits before verification. Do not return to or save over the '
                                       'protected source. Completion remains subject to evidence checks.')
                    elif self.plugins is not None:
                        try:
                            plugin_hint = self.plugins.hint_for_prompt(
                                state.foreground_window_title, goal)
                        except Exception as e:
                            self.on_log(f"[plugins] hint lookup failed: {e}")

                    plugin_hint += '\n' + progress_hint + '\nINTERACTION POLICY (overrides shortcut recipes): ' + interaction.hint()
                    action = self.reasoning.decide_next_action(
                        goal=goal,
                        screen_state=state,
                        history=history,
                        plan=plan,
                        current_step_idx=None,
                        memory_context=memory_context,
                        plugin_hint=plugin_hint,
                    )
            except Exception as e:
                self.on_log(f"   ❌ Reasoning error: {e}")
                consecutive_failures += 1
                if consecutive_failures >= self.max_consecutive_failures:
                    duration = time.monotonic() - start_time
                    self._record_history(task_id, goal, plan, history, False,
                                          f"Reasoning errored {consecutive_failures}x", step, duration)
                    return TaskResult(task_id, False, f"Reasoning failed {consecutive_failures}x in a row",
                                       step, plan, history, duration)
                time.sleep(2)
                continue

            if self.is_stopped():
                break
            self.on_log(
                f"   → {action.action_type.upper()}"
                f"{f' tag=[{action.target_tag}]' if action.target_tag else ''}"
                f"{f' text={action.text!r}' if action.text else ''}"
            )
            self.on_log(f"   reasoning: {action.reasoning}")

            # Show the target highlighted in blue
            if action.target_tag is not None:
                self.on_perceive(state, action.target_tag)

            # Terminal states
            if action.action_type == "done":
                evidence = input_evidence.hint() or source_prerequisite or document_evidence.completion_prerequisite() or document_evidence.check(state)
                verified = False
                if not evidence:
                    observed = observation(state) + '\n' + document_evidence.completion_evidence()
                    verified, evidence = verify_completion(self.reasoning.provider, goal, observed, history,
                                                           proposed_answer=action.text or '',
                                                           requirements=requirements, assessments=assessments,
                                                           observation_history=ledger.prior_observations())
                if self.is_stopped():
                    break
                if not verified:
                    rejected_completions += 1
                    history.append({'action': 'completion_check', 'outcome': evidence, 'success': False,
                                    'requirements': {k: dict(v) for k, v in assessments.items()}})
                    self.on_log(f'Completion rejected: {evidence}')
                    if rejected_completions < 2:
                        consecutive_failures = 2
                        continue
                    action = NextAction('fail', text='Completion could not be verified: ' + evidence)
                else:
                    history.append({'action': 'completion_check', 'outcome': evidence, 'success': True,
                                    'requirements': {k: dict(v) for k, v in assessments.items()}})

            if action.action_type == "done":
                self._set_status("Completed")
                self.on_log(f"\n✅ {action.text or 'Goal achieved.'}")
                duration = time.monotonic() - start_time
                self._record_history(task_id, goal, plan, history, True,
                                      action.text or "Goal achieved.", step, duration)
                self.on_step_complete(step, action, ExecutionResult(True, "Done"))
                return TaskResult(task_id, True, action.text or "Goal achieved.", step, plan, history, duration)

            if action.action_type == "fail":
                if (isinstance(action.raw, dict) and action.raw.get('blocked_by') == 'invalid_key'
                        and not any(item.get('action') == 'invalid_action_retry' for item in history)
                        and not self.is_stopped()):
                    history.append({'action': 'invalid_action_retry', 'success': False,
                                    'outcome': action.text})
                    self.on_log('Rejected malformed key action; reobserving and allowing one corrected decision. No input executed.')
                    continue
                transient = any(name in action.text for name in ('ConnectError', 'ConnectTimeout', 'ReadTimeout', 'RemoteProtocolError'))
                if transient and transport_failures < 2 and not self.is_stopped():
                    transport_failures += 1
                    self.on_log(f'Transient inference failure; bounded retry {transport_failures}/2. No desktop action repeated.')
                    history.append({'action': 'inference_retry', 'outcome': action.text, 'success': False})
                    time.sleep(1)
                    continue
                self._set_status("Failed")
                self.on_log(f"\n❌ {action.text or 'Agent gave up.'}")
                history.append({'action': 'agent_fail', 'outcome': action.text or 'Agent gave up.',
                                'success': False, 'observation': observation(state)[:2400],
                                'reasoning': action.reasoning,
                                'timestamp': datetime.now(timezone.utc).isoformat()})
                duration = time.monotonic() - start_time
                self._record_history(task_id, goal, plan, history, False,
                                      action.text or "Agent gave up.", step, duration)
                self.on_step_complete(step, action, ExecutionResult(False, "Failed"))
                return TaskResult(task_id, False, action.text or "Agent gave up.", step, plan, history, duration)

            # SAFETY CHECK
            if self.is_stopped():
                break
            signature = f'{action.action_type}:{action.target_tag}:{action.text}'
            blocked = attempts.check(observation(state), desktop_attempt_signature(state, action))
            input_block = input_evidence.restriction(action)
            if input_block:
                blocked = input_block
                attempts.blocked += 1
            redundant_creation = document_evidence.redundant_creation(action)
            duplicate = (document_evidence.new_document_guard(state, action, history)
                         or redundant_creation
                         or redundant_calculator_input(state, action, history)
                         or misplaced_save_action(state, action, history)
                         or unfinished_replacement_save(state, action, goal)
                         or (misplaced_document_edit(state, action, goal, source_prerequisite)
                             if source_prerequisite or not output_recovery else '')
                         or protected_source_save(state, action, goal))
            if duplicate:
                blocked = duplicate
                attempts.blocked += 1
            interaction_block = interaction.restriction(action, state)
            if interaction_block:
                blocked = interaction_block
                attempts.blocked += 1
            if blocked:
                record = {'action': signature, 'outcome': blocked, 'success': False}
                if redundant_creation and blocked == redundant_creation:
                    record['record_type'] = 'verified_redundancy_block'
                history.append(record)
                consecutive_failures = 2
                self.on_log(blocked)
                if attempts.blocked >= 3:
                    break
                continue
            assessment = self.safety.assess(action, screen_state=state)
            # Model risk flags alone miss generic Yes/Send buttons and Enter.
            target = next((e for e in state.elements if e.tag == action.target_tag), None)
            target_name = target.name.lower() if target else ''
            observed_low = observation(state).lower()
            if ((action.action_type in ('click', 'double_click') and
                 (target_name in ('send', 'delete', 'overwrite', 'purchase') or
                  (target_name in ('replace', 'replace all') and not requested_editor_replacement(state, action, goal)) or
                  (target_name == 'yes' and any(w in observed_low for w in ('replace', 'overwrite', 'already exists'))))) or
                (action.action_type == 'key' and action.text.lower() in ('enter', 'return') and
                 not open_file_dialog(state) and
                 any(w in observed_low for w in ('whatsapp', 'replace', 'overwrite', 'already exists')))):
                assessment = RiskAssessment(RiskLevel.HIGH, 'Observed send/overwrite/destructive control requires confirmation.', requires_confirmation=True)
            self.on_log(f"🛡  Risk: {assessment.level.value} — {assessment.reason}")

            if assessment.requires_confirmation:
                self._set_status(f"Awaiting confirmation ({assessment.level.value})")
                approved = self.on_confirm(action, assessment)
                if self.is_stopped():
                    break
                # SEC-5: log the decision either way, with a timestamp.
                if self.audit_log is not None:
                    try:
                        self.audit_log.record_confirmation(
                            action.action_type, assessment.level.value,
                            approved, assessment.reason)
                    except Exception as e:
                        self.on_log(f"[audit] could not record confirmation: {e}")
                if not approved:
                    self.on_log("   🚫 User cancelled.")
                    history.append({"action": str(action.action_type), "outcome": "User cancelled"})
                    duration = time.monotonic() - start_time
                    self._record_history(task_id, goal, plan, history, False, "User cancelled", step, duration)
                    return TaskResult(task_id, False, "User cancelled high-risk action.",
                                       step, plan, history, duration)

                # The modal belongs to PunkOS and steals focus from the target.
                # Restore the exact observed window and discard stale targets.
                if not self.controller.restore_focus(state):
                    message = 'Could not restore the observed window after confirmation; no input sent.'
                    detail = getattr(self.controller, 'focus_restore_detail', '')
                    if isinstance(detail, str) and detail:
                        message += ' ' + detail
                    history.append({'action': 'confirmation_focus', 'outcome': message, 'success': False,
                                    'timestamp': datetime.now(timezone.utc).isoformat()})
                    self.on_log(message)
                    self._record_history(task_id, goal, plan, history, False, message, step, time.monotonic() - start_time)
                    return TaskResult(task_id, False, message, step, plan, history, time.monotonic() - start_time)
                fresh_state = self.perception.perceive()
                if (fresh_state.foreground_window_handle != state.foreground_window_handle or
                        observation(fresh_state) != observation(state)):
                    history.append({'action': 'confirmation', 'outcome': 'Target changed during confirmation; reobserve and decide again.', 'success': False})
                    continue
                state = fresh_state

            if self.is_stopped():
                break

            # EXECUTE
            self._set_status(f"Executing (step {step})")
            self.on_log("✋ Executing...")
            result = self.controller.execute(action, state)
            input_evidence.record(action, state, result)
            self.on_log(f"   {'✅' if result.success else '❌'} {result.message}")

            # Record in history (in-memory + on-disk)
            history.append({
                "action": f"{action.action_type} tag={action.target_tag} text={action.text!r}",
                "outcome": result.message,
                "success": result.success,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "observation": observation(state)[:2400],
                "reasoning": action.reasoning,
            })
            self.on_step_complete(step, action, result)

            # Failure tracking
            if result.success:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= self.max_consecutive_failures:
                    self._set_status("Failed")
                    self.on_log(f"\n❌ Failed {consecutive_failures}x in a row. Stopping.")
                    duration = time.monotonic() - start_time
                    self._record_history(task_id, goal, plan, history, False,
                                          f"Failed {consecutive_failures}x in a row", step, duration)
                    return TaskResult(task_id, False, f"Failed {consecutive_failures}x in a row",
                                       step, plan, history, duration)

            time.sleep(self.step_delay)

        # Cancellation and recovery can end the loop before its step limit.
        self._set_status("Stopped" if self.is_stopped() else "Budget exhausted")
        duration = time.monotonic() - start_time
        message = self._stop_message if self.is_stopped() else 'Recovery or step budget exhausted; task is incomplete.'
        self.on_log(f"\n⏱  {message}")
        self._record_history(task_id, goal, plan, history, False, message, step, duration)
        return TaskResult(task_id, False, message, step, plan, history, duration)

    def _record_history(self, task_id, goal, plan, history, success, message, steps, duration, allow_learning=True):
        """Persist the episode (M11) and learn from it (M10).

        Every exit path in run() calls this, so it is the single place where
        end-of-task bookkeeping happens.
        """
        from .task_report import partial_report
        message = partial_report(message, success, plan, history)
        if self.history_store is not None:
            try:
                self.history_store.add_task(TaskRecord(
                    task_id=task_id,
                    goal=goal,
                    plan_summary=plan.summary if plan else None,
                    plan_json=plan.to_dict() if plan else None,
                    history_json=history,
                    success=success,
                    final_message=message,
                    steps_taken=steps,
                    duration_seconds=duration,
                    provider=self.reasoning.provider.name(),
                    created_at=datetime.now(timezone.utc).isoformat(),
                    metrics=self._metrics.snapshot() if getattr(self, '_metrics', None) is not None else {},
                ))
            except Exception as e:
                self.on_log(f"[history] failed to save: {e}")

        # M10: fold this run into procedural memory.
        self._last_learning = None
        if self.learner is not None and allow_learning:
            try:
                corrections = []
                if self.memory is not None:
                    corrections = self.memory.corrections_for_task(task_id)
                outcome = self.learner.learn_from_task(
                    task_id=task_id, goal=goal, history=history,
                    success=success, corrections=corrections,
                )
                self._last_learning = outcome.message if outcome.learned else None
            except Exception as e:
                self.on_log(f"[learning] failed: {e}")

    # ------------------------------------------------------------------
    # Sub-agent execution paths
    # ------------------------------------------------------------------

    def _run_web_task(self, task_id: str, goal: str,
                      routing: RoutingDecision, start_time: float) -> TaskResult:
        """M7: hand the goal to the Playwright agent.

        The web agent needs a concrete instruction set rather than a free-form
        goal, so we ask the LLM to turn the goal into a plan and execute the
        navigation/fill/scrape primitives it names. Anything the web agent
        cannot express falls back to reporting failure rather than silently
        doing something else.
        """
        self._set_status("Web agent")
        history: list[dict] = []

        from .web_agent import WebAgent
        if isinstance(self.web_agent, WebAgent):
            template = self.web_agent
            self.web_agent = WebAgent(browser=template.browser_name, headless=template.headless,
                max_tabs=template.max_tabs, navigation_timeout_ms=template.navigation_timeout_ms,
                channel=('chrome' if 'chrome' in goal.lower() else template.channel), download_dir=str(template.download_dir), on_log=self.on_log,
                audit_log=self.audit_log)
        if not self.web_agent.is_running:
            self.on_log("🌐 Launching browser...")
            result = self.web_agent.start()
            self.on_log(f"   {'✅' if result.success else '❌'} {result.message}")
            if not result.success:
                duration = time.monotonic() - start_time
                self._record_history(task_id, goal, None, history, False,
                                     result.message, 0, duration)
                return TaskResult(task_id, False, result.message, 0, None, history,
                                  duration, routing.mode.value, routing.reason)

        from .web_tasks import run_web_task
        def confirm(action, target, text, row):
            request = NextAction('web_' + action, text=text,
                                 reasoning=f"Target: {row}", is_high_risk=True)
            assessment = RiskAssessment(RiskLevel.HIGH, 'Browser interaction may submit or modify data.', requires_confirmation=True)
            return self.on_confirm(request, assessment)
        try:
            success, message, history = run_web_task(
                self.web_agent, self.reasoning.provider, goal, self.max_steps,
                self.is_stopped, confirm, self.on_log)
        except Exception as exc:
            success, message = False, f'Web task failed: {type(exc).__name__}: {exc}'
        finally:
            self.web_agent.close()
        if self.is_stopped():
            success, message = False, self._stop_message
        duration = time.monotonic() - start_time
        self._set_status('Completed' if success else 'Failed')
        self._record_history(task_id, goal, None, history, success, message, len(history), duration)
        return TaskResult(task_id, success, message, len(history), None, history,
                          duration, routing.mode.value, routing.reason)

    def _run_code_task(self, task_id: str, goal: str,
                       routing: RoutingDecision, start_time: float) -> TaskResult:
        """M9 FE-1 / OB-3: ask the LLM for a script, validate it, run it sandboxed."""
        from .sandbox import build_code_prompt

        self._set_status("Generating script")
        self.on_log("💻 Asking the model for a Python script...")
        history: list[dict] = []

        authorized = self.safety.authorized_dirs
        response = self.reasoning.provider.call_with_schema(
            system_prompt=build_code_prompt(authorized),
            user_text=f"TASK: {goal}\n\nWrite the script.",
            json_schema={
                "type": "object",
                "properties": {
                    "script": {"type": "string",
                               "description": "The complete Python script."},
                    "explanation": {"type": "string",
                                    "description": "One sentence on what it does."},
                },
                "required": ["script", "explanation"],
            },
            schema_name="python_script",
            schema_description="A safe Python script for a bulk file operation.",
        )

        if self.is_stopped():
            duration = time.monotonic() - start_time
            self._record_history(task_id, goal, None, history, False, self._stop_message, 0, duration)
            return TaskResult(task_id, False, self._stop_message, 0, None, history, duration,
                              routing.mode.value, routing.reason)
        if response.error or not response.parsed_json:
            message = f"Could not generate a script: {response.error or 'no JSON returned'}"
            self.on_log(f"   ❌ {message}")
            duration = time.monotonic() - start_time
            self._record_history(task_id, goal, None, history, False, message, 0, duration)
            return TaskResult(task_id, False, message, 0, None, history, duration,
                              routing.mode.value, routing.reason)

        script = str(response.parsed_json.get("script", ""))
        explanation = str(response.parsed_json.get("explanation", ""))
        self.on_log(f"   📝 {explanation}")

        # A bulk file script is inherently high-risk — always confirm it, and
        # show the user the actual code they are approving.
        preview_action = NextAction(
            action_type="code_execute",
            text=script[:400] + ("..." if len(script) > 400 else ""),
            reasoning=explanation,
            is_high_risk=True,
        )
        assessment = self.safety.assess(preview_action)
        self._set_status("Awaiting confirmation (HIGH)")
        approved = self.on_confirm(preview_action, assessment)
        if self.is_stopped():
            duration = time.monotonic() - start_time
            self._record_history(task_id, goal, None, history, False, self._stop_message, 0, duration)
            return TaskResult(task_id, False, self._stop_message, 0, None, history, duration,
                              routing.mode.value, routing.reason)
        if self.audit_log is not None:
            try:
                self.audit_log.record_confirmation("code_execute", "HIGH",
                                                   approved, explanation)
            except Exception:
                pass
        if not approved:
            message = "User cancelled the generated script."
            self.on_log(f"   🚫 {message}")
            duration = time.monotonic() - start_time
            self._record_history(task_id, goal, None, history, False, message, 0, duration)
            return TaskResult(task_id, False, message, 0, None, history, duration,
                              routing.mode.value, routing.reason)

        self._set_status("Executing in sandbox")
        result = self.sandbox.execute(script)
        history.append({
            "action": f"code_execute sha={result.script_sha256[:12]}",
            "outcome": (result.output or result.error or "")[:300],
            "success": result.success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        if result.success:
            self.on_log(f"   ✅ Script finished in {result.execution_time:.2f}s")
            if result.output:
                self.on_log(f"   output: {result.output.strip()[:300]}")
            message = result.output.strip() or "Script completed successfully."
        else:
            self.on_log(f"   ❌ {result.error[:300]}")
            message = result.error or "Script failed."

        self._set_status("Completed" if result.success else "Failed")
        duration = time.monotonic() - start_time
        self._record_history(task_id, goal, None, history, result.success,
                             message, 1, duration)
        return TaskResult(task_id, result.success, message, 1, None, history, duration,
                          routing.mode.value, routing.reason,
                          getattr(self, "_last_learning", None))

    @staticmethod
    def _extract_url(goal: str) -> Optional[str]:
        import re
        match = re.search(r"https?://\S+", goal)
        if match:
            return match.group(0).rstrip(".,)")
        match = re.search(r"\b((?:www\.)?[\w\-]+\.(?:com|org|net|io|edu|gov|co\.uk|pk))\b",
                          goal, re.I)
        return match.group(1) if match else None

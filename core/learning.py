"""
M10: Continuous Learning Subsystem — the Improver (SDS 1.7.10, Algorithm 6).

After every task the agent asks: "was that worth remembering?" A successful run
becomes a procedure in M11's procedural memory. A run of a task we have seen
before merges into the existing procedure, reweighting each step by how often it
has actually worked. User corrections overwrite the step they corrected and are
flagged so they win in future.

FE-1: incorporate corrections and successful demonstrations
FE-2: optimise repeated tasks by memorising the shortest path that works

What "learning" means here (and does not)
----------------------------------------
No gradients, no fine-tuning. This is frequency-weighted case-based reasoning:
store what worked, retrieve it by embedding similarity, prefer the steps with
the best track record. That is the honest description, and it is what Algorithm
6 in the SDS specifies. The benefit is real and measurable — a remembered
procedure skips the planning LLM call entirely, which on CPU is the single most
expensive part of a task.
"""
from __future__ import annotations

import re
import ast
import copy
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Sequence

from .memory import Correction, LearnedWorkflow, ProceduralMemory, WorkflowStep

# Steps that carry no procedural information — remembering them teaches nothing.
NON_PROCEDURAL_ACTIONS = {"wait", "done", "fail", "completion_check", "inference_retry"}

# Parses the history entries the planner writes:
#   "key tag=None text='win+r'"
_HISTORY_RE = re.compile(
    r"^(?P<action>\w+)\s+tag=(?P<tag>\S+)\s+text=(?P<text>.*)$"
)


@dataclass
class LearningOutcome:
    learned: bool
    action: str                      # "created" | "merged" | "skipped"
    workflow_id: Optional[str] = None
    similarity: Optional[float] = None
    message: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_history_entry(entry: dict) -> Optional[WorkflowStep]:
    """Turn one planner history row into a procedural step."""
    raw = str(entry.get("action", "")).strip()
    if not raw:
        return None

    match = _HISTORY_RE.match(raw)
    if match:
        action = match.group("action")
        text = match.group("text").strip()
        try:
            decoded = ast.literal_eval(text)
            if not isinstance(decoded, str):
                return None
            text = decoded
        except (ValueError, SyntaxError):
            return None
    else:
        action = raw.split()[0]
        text = ""

    if action in NON_PROCEDURAL_ACTIONS:
        return None

    return WorkflowStep(
        action_type=action,
        text=text,
        target_hint=str(entry.get("target_hint", "")),
        success_weight=1.0 if entry.get("success", True) else 0.0,
    )


def extract_steps(history: Sequence[dict]) -> list[WorkflowStep]:
    """Keep only the steps that actually worked — failures teach the wrong path."""
    steps = []
    for entry in history:
        if entry.get("success") is not True:
            continue
        step = parse_history_entry(entry)
        if step is not None:
            steps.append(step)
    return steps


def derive_workflow_name(goal: str) -> str:
    cleaned = re.sub(r"\s+", " ", goal).strip()
    return cleaned[:60] if len(cleaned) <= 60 else cleaned[:57] + "..."


class ContinuousLearner:
    """Implements SDS Algorithm 6 over ProceduralMemory."""

    def __init__(
        self,
        memory: ProceduralMemory,
        similarity_threshold: float = 0.85,
        on_log: Optional[Callable[[str], None]] = None,
    ):
        self.memory = memory
        self.similarity_threshold = similarity_threshold
        self.on_log = on_log or (lambda _m: None)

    # ------------------------------------------------------------------

    def learn_from_task(
        self,
        task_id: str,
        goal: str,
        history: Sequence[dict],
        success: bool,
        corrections: Optional[Sequence[Correction]] = None,
        force: bool = False,
    ) -> LearningOutcome:
        """Algorithm 6, end to end.

        `force=True` is the "Save as Template" path (FR7.3): the user asked for
        it, so we store it even if the run failed or was trivially short.
        """
        corrections = list(corrections or [])
        if not force:
            checks = [row for row in history if row.get('action') == 'completion_check']
            if not checks or checks[-1].get('success') is not True or history[-1] is not checks[-1]:
                return LearningOutcome(False, 'skipped', message='No final successful completion check; run retained only in history.')
            if any(row.get('success') is not True and parse_history_entry(row) is not None for row in history):
                return LearningOutcome(False, 'skipped', message='Run contains failed or uncertain actions; do not learn a shortened recovery trace.')
        if not force and any(str(row.get('action', '')).split(' ', 1)[0] in
                             {'navigate', 'search', 'download'} for row in history):
            # Browser observations/targets are session-bound, and the current
            # desktop procedure format loses web URLs and target identity.
            return LearningOutcome(False, 'skipped', message='Web task retained in history; automatic reusable-procedure learning is not supported for browser traces.')
        steps = extract_steps(history)

        if not force:
            if not success:
                return LearningOutcome(False, "skipped",
                                       message="Task did not succeed — nothing learned")
            if len(steps) < 2:
                return LearningOutcome(False, "skipped",
                                       message="Too few steps to be a reusable procedure")

        if not steps:
            return LearningOutcome(False, "skipped",
                                   message="No procedural steps to store")

        # Step 1-2: does something similar already exist?
        match = None if force else self.memory.find_similar_workflow(goal, threshold=self.similarity_threshold)
        if match:
            candidate = match[0]
            # Similar goals are not interchangeable (paths, recipients and
            # literal contents matter). Preserve each demonstrated sequence.
            signature = lambda values: [(s.action_type, s.text, s.target_hint) for s in values]
            if candidate.is_template or candidate.trigger_pattern != goal or signature(candidate.steps) != signature(steps):
                match = None

        if match:
            existing, score = match
            merged = self._merge_steps(existing.steps, steps)
            merged = self._apply_corrections(merged, corrections)

            existing.steps = merged
            existing.execution_count += 1
            existing.success_count += 1 if success else 0
            existing.last_used = _now()
            if force:
                existing.is_template = True
            self.memory.save_workflow(existing)

            self.on_log(
                f"[learning] merged into '{existing.name}' "
                f"(similarity {score:.0%}, now {existing.success_rate:.0%} "
                f"over {existing.execution_count} runs)"
            )
            return LearningOutcome(True, "merged", existing.workflow_id, score,
                                   f"Merged into existing workflow '{existing.name}'")

        # Step 4: nothing similar — create a new procedure.
        steps = self._apply_corrections(steps, corrections)
        workflow = LearnedWorkflow(
            workflow_id=uuid.uuid4().hex[:12],
            name=derive_workflow_name(goal),
            trigger_pattern=goal,
            steps=steps,
            execution_count=1,
            success_count=1 if success else 0,
            last_used=_now(),
            created_at=_now(),
            is_template=force,
            source_task_id=task_id,
        )
        self.memory.save_workflow(workflow)
        self.on_log(f"[learning] learned new procedure '{workflow.name}' "
                    f"({len(steps)} steps)")
        return LearningOutcome(True, "created", workflow.workflow_id, None,
                               f"Learned new workflow '{workflow.name}'")

    # ------------------------------------------------------------------

    @staticmethod
    def _step_key(step: WorkflowStep) -> tuple[str, str]:
        return (step.action_type, step.text.strip().lower())

    def _merge_steps(
        self,
        existing: Sequence[WorkflowStep],
        observed: Sequence[WorkflowStep],
    ) -> list[WorkflowStep]:
        """Preserve observed order and repetitions; weight matching positions only."""
        result = copy.deepcopy(list(observed))
        for index, step in enumerate(result):
            if index < len(existing):
                prior = existing[index]
                if (prior.action_type, prior.text, prior.target_hint) == (step.action_type, step.text, step.target_hint):
                    step.success_weight = prior.success_weight + 1
                    step.from_correction = prior.from_correction
        return result

    @staticmethod
    def _apply_corrections(
        steps: list[WorkflowStep],
        corrections: Sequence[Correction],
    ) -> list[WorkflowStep]:
        """Algorithm 6 step 5: the user's correction replaces the step it fixed."""
        for correction in corrections:
            idx = correction.step_index
            replacement = WorkflowStep(
                action_type="corrected",
                text=correction.corrected_action,
                from_correction=True,
                success_weight=2.0,          # outrank anything learned by observation
            )
            # Parse "action text" out of the corrected action if it looks structured.
            parts = correction.corrected_action.split(None, 1)
            if parts and parts[0] in {
                "click", "double_click", "right_click", "type", "key", "scroll", "drag"
            }:
                replacement.action_type = parts[0]
                replacement.text = parts[1] if len(parts) > 1 else ""

            if 0 <= idx < len(steps):
                steps[idx] = replacement
            else:
                steps.append(replacement)
        return steps

    # ------------------------------------------------------------------

    def suggest_for_goal(self, goal: str) -> Optional[tuple[LearnedWorkflow, float]]:
        """Retrieve a remembered procedure to inject before planning."""
        return self.memory.find_similar_workflow(goal, threshold=self.similarity_threshold)

    def repeated_pattern_candidates(self, min_executions: int = 3,
                                    min_success_rate: float = 0.7) -> list[LearnedWorkflow]:
        """Event-response table: workflows worth proposing to the user as templates."""
        return [
            w for w in self.memory.list_workflows(limit=200)
            if w.execution_count >= min_executions
            and w.success_rate >= min_success_rate
            and not w.is_template
        ]

    def proactive_suggestion(self) -> Optional[str]:
        """M2 FE-2: 'I notice you do this often — shall I automate it?'"""
        candidates = self.repeated_pattern_candidates()
        if not candidates:
            return None
        top = max(candidates, key=lambda w: w.execution_count)
        return (
            f"You've run \"{top.name}\" {top.execution_count} times with a "
            f"{top.success_rate:.0%} success rate. Save it as a reusable template?"
        )

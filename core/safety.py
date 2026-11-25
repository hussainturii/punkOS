"""
M9: Safety and Verification System (The Guardrails)

Implements SDS 1.7.9:
  FE-1  Secure sandboxed execution      -> core/sandbox.py
  FE-2  Human-in-the-loop confirmation  -> assess() + confirm callback
  FE-3  Boundary enforcement            -> is_path_allowed() / authorize_paths()

Risk classification follows SDS Algorithm 5 (Risk Assessment):
  HIGH   - destructive verb, >10 affected files, or a system-critical path
  MEDIUM - mutating verb, or 2-10 affected files
  LOW    - everything else

Design note on keyword matching
-------------------------------
Keywords are matched on WORD BOUNDARIES against the action's *literal* fields
(action_type + text), never against the LLM's free-text `reasoning` narration.
Substring matching over narration produced constant false positives — 'drop'
fired on "click the dropdown", 'pay' on "payment page", 'reset' on "preset".
Every such false positive halts an autonomous run on a confirmation modal, so
the matcher is deliberately strict. The LLM's own `is_high_risk` self-flag is
still honoured as an independent signal.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence

from .reasoning import NextAction
from .agent_checks import open_file_dialog


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class RiskAssessment:
    level: RiskLevel
    reason: str
    requires_confirmation: bool
    # FR2.3: diff-style preview rows -> (change_kind, detail)
    # change_kind is one of "add" (green), "delete" (red), "modify" (amber)
    preview: list[tuple[str, str]] = field(default_factory=list)
    affected_paths: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """FR2.2 / USE-3: plain-language description for the confirmation modal."""
        return self.reason


# SDS Algorithm 5 step 2 — destructive / irreversible verbs
HIGH_RISK_KEYWORDS = {
    "delete", "remove", "uninstall", "format", "wipe", "erase", "purge",
    "send", "submit", "purchase", "pay", "transfer", "shutdown", "restart",
    "drop", "destroy", "kill", "reset",
}

# SDS Algorithm 5 step 3 — mutating but recoverable verbs
MEDIUM_RISK_KEYWORDS = {
    "move", "rename", "overwrite", "modify", "replace", "save", "update",
}

# Hotkeys that are destructive regardless of surrounding wording.
DESTRUCTIVE_HOTKEYS = {"delete", "del", "shift+delete", "ctrl+shift+delete"}

# SDS 4.3.3 — hardcoded blocklist, applied regardless of the user's allow list.
DEFAULT_BLOCKED_PATHS = (
    r"C:\Windows",
    r"C:\Program Files",
    r"C:\Program Files (x86)",
    r"C:\ProgramData",
    "/etc", "/bin", "/usr", "/boot", "/dev", "/sys",
)

_WORD_RE_CACHE: dict[str, re.Pattern] = {}


def _matches_word(keyword: str, text: str) -> bool:
    """Whole-word, case-insensitive match. 'drop' does NOT match 'dropdown'."""
    pattern = _WORD_RE_CACHE.get(keyword)
    if pattern is None:
        pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
        _WORD_RE_CACHE[keyword] = pattern
    return bool(pattern.search(text))


def _first_match(keywords: set[str], text: str) -> Optional[str]:
    for kw in sorted(keywords):
        if _matches_word(kw, text):
            return kw
    return None


# Default confirmation: console-based.
def cli_confirm(action: NextAction, assessment: RiskAssessment) -> bool:
    print()
    print("=" * 60)
    print(f"!  {assessment.level.value} RISK ACTION")
    print("=" * 60)
    print(f"Action:    {action.action_type}")
    if action.target_tag is not None:
        print(f"Target:    tag [{action.target_tag}]")
    if action.text:
        print(f"Text:      {action.text!r}")
    print(f"Reasoning: {action.reasoning}")
    print(f"Risk:      {assessment.reason}")
    if assessment.preview:
        print("Changes:")
        for kind, detail in assessment.preview:
            marker = {"add": "+", "delete": "-", "modify": "~"}.get(kind, "?")
            print(f"  {marker} {detail}")
    print("=" * 60)
    ans = input("Type 'yes' to proceed, anything else to cancel: ").strip().lower()
    return ans == "yes"


class SafetySystem:
    def __init__(
        self,
        authorized_dirs: Sequence[str],
        blocked_paths: Sequence[str] = (),
        confirm_medium_risk: bool = True,
        audit_log=None,
    ):
        self.authorized_dirs = [self._norm(p) for p in authorized_dirs]
        # User-supplied blocklist is merged with the hardcoded one (SDS 4.3.3):
        # the hardcoded entries apply even if config.json omits them.
        merged = list(blocked_paths) + list(DEFAULT_BLOCKED_PATHS)
        self.blocked_paths = [self._norm(p) for p in merged]
        self.confirm_medium_risk = confirm_medium_risk
        self.audit_log = audit_log

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _norm(path: str) -> str:
        """Expand ~, resolve symlinks, then normalise case/separators.

        expanduser() lets config.json ship portable paths like "~/Documents"
        instead of a hardcoded username, so the same config works for every
        team member and on every machine.
        realpath() defeats directory-traversal via symlink (SDS 4.3.3).
        normcase() makes the comparison correct on case-insensitive Windows.
        """
        expanded = os.path.expanduser(path)
        return os.path.normcase(os.path.abspath(os.path.realpath(expanded)))

    @staticmethod
    def _is_within(child: str, parent: str) -> bool:
        """True if `child` is `parent` or lives underneath it.

        Uses commonpath rather than startswith so that 'C:\\Users\\bobby' is
        NOT treated as living inside 'C:\\Users\\bob'.
        """
        try:
            return os.path.commonpath([child, parent]) == parent
        except ValueError:
            # Different drives on Windows -> no relationship.
            return False

    def is_path_allowed(self, path: str) -> tuple[bool, str]:
        """SDS FE-3 / SEC-2. Blocklist wins over the allow list, always."""
        try:
            abs_path = self._norm(path)
        except Exception as e:
            return False, f"Path resolution error: {e}"

        for blocked in self.blocked_paths:
            if self._is_within(abs_path, blocked):
                return False, f"Path is in a blocked system directory: {blocked}"

        for allowed in self.authorized_dirs:
            if self._is_within(abs_path, allowed):
                return True, f"Path is under authorized directory: {allowed}"

        return False, "Path is not under any authorized directory"

    def authorize_paths(self, paths: Sequence[str]) -> tuple[bool, list[str]]:
        """Check a batch of paths. Returns (all_allowed, list_of_rejection_reasons)."""
        rejections: list[str] = []
        for p in paths:
            ok, reason = self.is_path_allowed(p)
            if not ok:
                rejections.append(f"{p}: {reason}")
        if rejections and self.audit_log is not None:
            self.audit_log.record(
                event="boundary_violation",
                detail="; ".join(rejections),
            )
        return (not rejections), rejections

    # ------------------------------------------------------------------
    # Risk assessment — SDS Algorithm 5
    # ------------------------------------------------------------------

    def assess(
        self,
        action: NextAction,
        affected_paths: Optional[Sequence[str]] = None,
        *, screen_state=None,
    ) -> RiskAssessment:
        affected = list(affected_paths or [])

        # -- Step 2a: system-critical path in scope -> HIGH, non-negotiable.
        for path in affected:
            ok, reason = self.is_path_allowed(path)
            if not ok:
                return RiskAssessment(
                    RiskLevel.HIGH,
                    f"Action touches a restricted path — {reason}",
                    requires_confirmation=True,
                    preview=[("modify", p) for p in affected],
                    affected_paths=affected,
                )

        # -- LLM self-flagging is an independent HIGH signal.
        if action.is_high_risk:
            return RiskAssessment(
                RiskLevel.HIGH,
                f"Model flagged this action as irreversible: {action.reasoning[:120]}",
                requires_confirmation=True,
                preview=self._build_preview(action, affected, "delete"),
                affected_paths=affected,
            )

        # -- Step 2b: bulk scope -> HIGH above 10 files.
        if len(affected) > 10:
            return RiskAssessment(
                RiskLevel.HIGH,
                f"Bulk operation affecting {len(affected)} files",
                requires_confirmation=True,
                preview=self._build_preview(action, affected, "modify"),
                affected_paths=affected,
            )

        # -- Step 2c: keyword scan over LITERAL fields only (never `reasoning`).
        literal = f"{action.action_type} {action.text}"
        # In an observed Open filename field, a single absolute path is an
        # identifier, not an instruction to replace/delete named directories.
        # Keep model flags, affected-path checks and actual key actions above.
        if screen_state is not None and action.action_type == 'type' and not affected:
            target = next((e for e in screen_state.elements if e.tag == action.target_tag), None)
            if ((screen_state.foreground_window_title.strip().casefold() == 'open' or open_file_dialog(screen_state))
                    and target is not None and target.is_enabled
                    and target.role in {'EditControl', 'ComboBoxControl'}
                    and target.name.strip().rstrip(':').casefold() == 'file name'
                    and re.fullmatch(r'[a-zA-Z]:[\\/][^<>:"|?*\r\n]+', action.text or '')):
                literal = action.action_type

        if action.action_type == "key" and action.text.strip().lower() in DESTRUCTIVE_HOTKEYS:
            return RiskAssessment(
                RiskLevel.HIGH,
                f"Destructive keystroke: {action.text!r}",
                requires_confirmation=True,
                affected_paths=affected,
            )

        hit = _first_match(HIGH_RISK_KEYWORDS, literal)
        if hit:
            return RiskAssessment(
                RiskLevel.HIGH,
                f"Action is a {hit!r} operation, which cannot be undone",
                requires_confirmation=True,
                preview=self._build_preview(action, affected, "delete"),
                affected_paths=affected,
            )

        # -- Step 3: medium triggers.
        if 2 <= len(affected) <= 10:
            return RiskAssessment(
                RiskLevel.MEDIUM,
                f"Operation affects {len(affected)} files",
                requires_confirmation=self.confirm_medium_risk,
                preview=self._build_preview(action, affected, "modify"),
                affected_paths=affected,
            )

        hit = _first_match(MEDIUM_RISK_KEYWORDS, literal)
        if hit:
            return RiskAssessment(
                RiskLevel.MEDIUM,
                f"Action modifies existing data ({hit!r})",
                requires_confirmation=self.confirm_medium_risk,
                preview=self._build_preview(action, affected, "modify"),
                affected_paths=affected,
            )

        # -- Step 4: nothing matched.
        return RiskAssessment(
            RiskLevel.LOW,
            "Routine interaction with no destructive indicators",
            requires_confirmation=False,
            affected_paths=affected,
        )

    @staticmethod
    def _build_preview(
        action: NextAction,
        affected: Sequence[str],
        kind: str,
    ) -> list[tuple[str, str]]:
        """FR2.3: rows for the modal's diff panel."""
        if affected:
            return [(kind, p) for p in affected]
        descriptor = action.text or f"tag [{action.target_tag}]"
        return [(kind, f"{action.action_type} -> {descriptor}")]

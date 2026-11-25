"""
Sub-agent routing and execution-mode selection.

Implements M3 FE-2 ("route tasks to the appropriate sub-agent") and M4 FE-3
("determine the optimal execution mode: DOM control, sandboxed code, or GUI").

This is the Fast Brain half of the Fast/Slow split (M3 FE-3). Routing runs
locally on keyword and shape heuristics in well under a millisecond, so the
expensive LLM call is only made once the mode is already known and the prompt
can be narrowed to that mode's action vocabulary. Asking the 3B model to both
choose a mode and plan within it measurably degrades both.

Modes
-----
DESKTOP  GUI control through the accessibility tree (M1 + M5). The default.
WEB      Playwright DOM control (M7). Chosen for URLs and browser verbs,
         because Chrome's UIA tree is the documented M1 blind spot.
CODE     Sandboxed Python (M9 FE-1). Chosen for bulk file work, where clicking
         once per file is hopeless — this is OB-3's minutes-to-milliseconds win.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ExecutionMode(Enum):
    FILE_OPERATION = "FILE_OPERATION"
    FILE_SEARCH = "FILE_SEARCH"
    DESKTOP = "DESKTOP"
    WEB = "WEB"
    CODE = "CODE"


@dataclass
class RoutingDecision:
    mode: ExecutionMode
    confidence: float
    reason: str

    def to_dict(self) -> dict:
        return {"mode": self.mode.value, "confidence": self.confidence, "reason": self.reason}


_URL_RE = re.compile(r"(https?://|www\.|\b\w+\.(com|org|net|io|edu|gov|co\.uk|pk)\b)", re.I)

_WEB_VERBS = (
    "browser", "website", "web page", "webpage", "chrome", "firefox", "edge",
    "google", "search online", "search the web", "fill the form", "fill out",
    "log in to", "sign in to", "scrape", "download from", "navigate to",
)

_CODE_VERBS = (
    "all files", "every file", "each file", "bulk", "batch",
    "rename all", "move all", "delete all", "copy all", "organise", "organize",
    "sort files", "clean up", "convert all",
)

# A count large enough that doing it by hand through the GUI is unreasonable.
_BULK_COUNT_RE = re.compile(r"\b(\d{2,})\s+(files?|documents?|photos?|images?|pdfs?)\b", re.I)

_FILE_CONTEXT = ("file", "folder", "directory", "downloads", "documents", "desktop")


def _contains_any(text: str, needles) -> Optional[str]:
    for n in needles:
        if n in text:
            return n
    return None


def route(goal: str) -> RoutingDecision:
    """Pick an execution mode for a goal. Never raises; defaults to DESKTOP."""
    if not goal or not goal.strip():
        return RoutingDecision(ExecutionMode.DESKTOP, 0.0, "Empty goal — defaulting to desktop")

    low = goal.lower()
    from .interaction_policy import mouse_requested
    if mouse_requested(goal):
        return RoutingDecision(ExecutionMode.DESKTOP, 1.0, 'User explicitly requested mouse interaction; respecting desktop mode.')

    from .file_operations import eligible
    if eligible(goal):
        return RoutingDecision(ExecutionMode.FILE_OPERATION, .95,
                               'Explicit filesystem operation with quoted paths; verified no-overwrite execution.')

    if re.search(r'\b(?:vs\s*code|visual studio code|notepad)\b', low):
        return RoutingDecision(ExecutionMode.DESKTOP, .95, 'User requested an editor workflow; URLs or code contents do not change the target application.')
    if re.search(r'\b(?:create|write|save)\b', low) and re.search(r'\b[\w-]+\.(?:py|txt|md|json|csv|js|html|css)\b', low):
        return RoutingDecision(ExecutionMode.DESKTOP, .9, 'Create/edit a named text file through the desktop editor.')

    # -- WEB: an explicit URL is the strongest signal there is.
    url_match = _URL_RE.search(goal)
    if url_match:
        return RoutingDecision(
            ExecutionMode.WEB, 0.95,
            f"Goal names a web address ({url_match.group(0)}) — DOM control is more "
            f"reliable than the browser's accessibility tree"
        )

    if (re.search(r'\b(?:find|search|locate|read)\b', low)
            and not re.search(r'\b(?:current|visible|active)\b|on.screen', low)
            and (re.search(r'\b(?:files?|documents?|notes?|reports?|invoices?|authorized folders)\b', low)
                 or re.search(r'\b[a-z]:[\\/].+\.(?:txt|md|log|csv|pdf|docx)\b', low))
            and not _contains_any(low, _WEB_VERBS)
            and not re.search(r'\b(?:delete|rename|move|create|overwrite|edit|execute)\b', low)):
        return RoutingDecision(ExecutionMode.FILE_SEARCH, .9, 'Read-only local document search in authorized folders.')

    # -- CODE: an explicit large count beats everything else.
    bulk = _BULK_COUNT_RE.search(goal)
    if bulk:
        return RoutingDecision(
            ExecutionMode.CODE, 0.9,
            f"Goal affects {bulk.group(1)} {bulk.group(2)} — a script is orders of "
            f"magnitude faster than clicking each one"
        )

    web_hit = _contains_any(low, _WEB_VERBS)
    code_hit = _contains_any(low, _CODE_VERBS)

    # Bulk phrasing only means CODE if it is actually about files. "delete all
    # the text in this document" is a desktop editing task, not a script.
    if code_hit and not _contains_any(low, _FILE_CONTEXT):
        code_hit = None

    if code_hit and web_hit:
        # Both fired — prefer WEB, since a browser task cannot be done by a
        # local file script at all, whereas bulk file work can fall back to GUI.
        return RoutingDecision(
            ExecutionMode.WEB, 0.6,
            f"Goal mentions both web ({web_hit!r}) and bulk file ({code_hit!r}) work; "
            f"routing to the web agent first"
        )

    if code_hit:
        return RoutingDecision(
            ExecutionMode.CODE, 0.8,
            f"Goal describes a bulk file operation ({code_hit!r}) — routing to the "
            f"sandboxed code path"
        )

    if web_hit:
        return RoutingDecision(
            ExecutionMode.WEB, 0.75,
            f"Goal describes browser work ({web_hit!r}) — routing to the Playwright agent"
        )

    return RoutingDecision(
        ExecutionMode.DESKTOP, 0.7,
        "No web or bulk-file indicators — using accessibility-tree desktop control"
    )

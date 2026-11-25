"""
M1: Visual Perception Engine
The "eyes" of PunkOS — captures screen, parses Windows accessibility tree,
and produces Set-of-Mark annotated screenshots for the LLM.
"""
from __future__ import annotations

import base64
from collections import deque
from itertools import islice
import io
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Optional
from .sensitive_content import contains_credentials

import mss
from PIL import Image, ImageDraw, ImageFont

# uiautomation is Windows-only.
if sys.platform == "win32":
    try:
        import uiautomation as auto
        UIA_AVAILABLE = True
    except ImportError:
        UIA_AVAILABLE = False
        auto = None
else:
    UIA_AVAILABLE = False
    auto = None


# Controls the agent can actually operate. These are always kept.
INTERACTIVE_CONTROL_TYPES = {
    "ButtonControl", "EditControl", "HyperlinkControl", "CheckBoxControl",
    "RadioButtonControl", "ComboBoxControl", "ListItemControl", "MenuItemControl",
    "TabItemControl", "TreeItemControl", "SliderControl", "SpinnerControl",
    "SplitButtonControl",
}

# Read-only controls. Useful as context (e.g. reading the Calculator display),
# but they are NOT click targets. They only fill leftover element budget.
#
# These used to sit in INTERACTIVE_CONTROL_TYPES. Because the tree walk is
# depth-first and stops at max_elements, a text-heavy window could exhaust the
# whole budget on static labels before the walk ever reached a real button —
# which is what made the agent "pick the wrong element".
CONTEXT_CONTROL_TYPES = {
    "TextControl", "DocumentControl", "ImageControl",
}

ALL_CAPTURED_CONTROL_TYPES = INTERACTIVE_CONTROL_TYPES | CONTEXT_CONTROL_TYPES


@dataclass
class UIElement:
    """A single element from the accessibility tree."""
    tag: int
    role: str
    name: str
    bounds: tuple[int, int, int, int]   # (left, top, right, bottom)
    is_enabled: bool
    automation_id: str = ""
    is_interactive: bool = True         # False for read-only context elements
    text: str = ""                     # UIA editor/document content, never password values
    is_editable: bool = False
    runtime_id: tuple = ()
    focus_only: bool = False

    @property
    def center(self) -> tuple[int, int]:
        l, t, r, b = self.bounds
        return ((l + r) // 2, (t + b) // 2)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScreenState:
    """Full perception output sent to the reasoning core."""
    screenshot_b64: str           # Annotated PNG, base64
    raw_screenshot_b64: str       # Raw PNG, base64
    elements: list[UIElement]
    foreground_window_title: str
    width: int
    height: int
    foreground_window_handle: int = 0
    foreground_process_id: int = 0
    sensitive_content_detected: bool = False
    accessibility_diagnostic: str = ''
    field_images: dict = field(default_factory=dict, repr=False)

    def elements_summary(self) -> list[dict]:
        """Compact representation for the LLM prompt."""
        return [
            {"tag": e.tag, "role": e.role, "name": e.name,
             "enabled": e.is_enabled, "interactive": e.is_interactive,
             "text": e.text}
            for e in self.elements
        ]

    def interactive_elements(self) -> list[UIElement]:
        return [e for e in self.elements if e.is_interactive]


class VisualPerceptionEngine:
    """Captures screen state for the reasoning core."""

    def __init__(self, max_elements: int = 40, max_dimension: int = 1280):
        self.max_elements = max_elements
        self.max_dimension = max_dimension
        # Budget reserved for read-only context elements (Calculator's display,
        # status text). The rest goes to elements the agent can actually click.
        self.context_budget = max(4, max_elements // 4)
        self._deep_windows = set()

    def capture_screen(self) -> Image.Image:
        """Capture the primary monitor as a PIL Image."""
        with mss.mss() as sct:
            monitor = sct.monitors[1]    # 0 = all monitors, 1 = primary
            raw = sct.grab(monitor)
            img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        return img

    def get_accessibility_tree(self, root=None) -> tuple[list[UIElement], str]:
        """Walk the UIA tree under the foreground window."""
        self._foreground_handle = 0
        self._foreground_process_id = 0
        self._tree_stats = {}
        if not UIA_AVAILABLE:
            return [], "[UIA unavailable — non-Windows or missing uiautomation]"

        interactive: list[UIElement] = []
        context: list[UIElement] = []
        try:
            fg = root if root is not None else auto.GetForegroundControl()
            if fg is None:
                return [], ""
            window_title = fg.Name or ""
            self._foreground_handle = int(fg.NativeWindowHandle or 0)
            try:
                self._foreground_process_id = int(fg.ProcessId or 0)
            except (AttributeError, TypeError, ValueError):
                self._foreground_process_id = 0
            window_key = (self._foreground_handle, self._foreground_process_id)
            scan_depth = 40 if window_key in self._deep_windows else 12
            self._walk_tree(fg, interactive, context, max_depth=scan_depth)
            first = self._tree_stats.copy()
            if scan_depth == 40:
                self._tree_stats['adaptive_depth'] = 40
            if first['depth_cutoffs'] and scan_depth == 12:
                # Electron/web editors can sit below many layout containers.
                # Retry only when the first scan actually hit the depth limit.
                interactive.clear()
                context.clear()
                self._walk_tree(fg, interactive, context, max_depth=40)
                self._tree_stats['deep_retry'] = True
                if window_key[0]:
                    if len(self._deep_windows) >= 16:
                        self._deep_windows.clear()
                    self._deep_windows.add(window_key)
            if self._tree_stats['node_limit'] or self._tree_stats['child_cutoffs']:
                region = self._focused_region(fg)
                if region is not None:
                    main_stats = self._tree_stats.copy()
                    focused, focused_context = [], []
                    self._walk_tree(region, focused, focused_context, max_depth=20, max_nodes=256)
                    focus_stats = self._tree_stats
                    self._tree_stats = main_stats
                    self._tree_stats['focus_retry'] = True
                    self._tree_stats['focus_visited'] = focus_stats['visited']
                    self._tree_stats['focus_truncated'] = bool(focus_stats['node_limit'] or
                        focus_stats['depth_cutoffs'] or focus_stats['child_cutoffs'])
                    # A crowded focus neighborhood can itself exhaust the scan.
                    # Capture the actual focused editor directly, after proving
                    # its ancestry, so sibling lists cannot crowd it out.
                    anchor = self._focused_editor(fg)
                    anchored = []
                    if anchor is not None:
                        saved_stats = self._tree_stats
                        self._walk_tree(anchor, anchored, [], max_depth=4, max_nodes=16)
                        self._tree_stats = saved_stats
                        self._tree_stats['focused_editor_captured'] = bool(anchored)
                    # Deduplicate by observed identity, never by label alone.
                    merged, seen = [], set()
                    focus_budget = max(1, self.max_elements * 3 // 4)
                    focused.sort(key=lambda e: not e.is_editable)
                    for element in anchored + focused[:focus_budget] + interactive + focused[focus_budget:]:
                        identity = (element.role, element.name, element.bounds, element.automation_id)
                        if identity not in seen:
                            merged.append(element)
                            seen.add(identity)
                    interactive[:] = merged[:self.max_elements]
                    context[:] = focused_context + context
        except Exception as e:
            print(f"[Perception] UIA walk failed: {e}")
            return [], ""

        # Dialog controls precede file rows; read-only context fills what is
        # left. Assign tags only after selection so prompts and annotations agree.
        kept = interactive[: self.max_elements]
        remaining = self.max_elements - len(kept)
        if remaining > 0:
            kept += context[: min(remaining, self.context_budget)]

        for i, el in enumerate(kept, start=1):
            el.tag = i
        return kept, window_title

    @staticmethod
    def _focused_editor(foreground):
        """Return only a writable focus node belonging to this exact window."""
        try:
            node = auto.GetFocusedControl()
            candidate = node
            if node.ControlTypeName not in {'EditControl', 'ComboBoxControl'}:
                return None
            if not node.IsEnabled or node.IsPassword or node.IsOffscreen:
                return None
            root_id = tuple(foreground.GetRuntimeId())
            if not root_id:
                return None
            seen = set()
            for _ in range(64):
                identity = tuple(node.GetRuntimeId())
                if identity == root_id:
                    return candidate
                if not identity or identity in seen:
                    return None
                seen.add(identity)
                node = node.GetParentControl()
        except Exception:
            return None
        return None

    @staticmethod
    def _focused_region(foreground):
        """Find a small focus neighborhood only after proving window ancestry."""
        try:
            root_id = tuple(foreground.GetRuntimeId())
            if not root_id:
                return None
            chain, seen = [], set()
            node = auto.GetFocusedControl()
            for _ in range(64):
                identity = tuple(node.GetRuntimeId())
                if identity == root_id:
                    if not chain:
                        return None
                    # Prefer an observed dialog/window ancestor. Otherwise use
                    # a bounded neighborhood, stopping before the whole page.
                    for ancestor in chain[1:]:
                        if ancestor.ControlTypeName == 'WindowControl':
                            return ancestor
                        if ancestor.ControlTypeName == 'DocumentControl':
                            break
                    region = chain[0]
                    for ancestor in chain[1:5]:
                        if ancestor.ControlTypeName == 'DocumentControl':
                            break
                        region = ancestor
                    return region
                if not identity or identity in seen:
                    return None
                seen.add(identity)
                chain.append(node)
                node = node.GetParentControl()
        except Exception:
            return None
        return None

    @staticmethod
    def _is_editable_document(node):
        """Use explicit UIA writability, never a document's name or app title."""
        try:
            if node.ControlTypeName != 'DocumentControl' or node.IsPassword:
                return False
            pattern = node.GetTextPattern()
            value = pattern.DocumentRange.GetAttributeValue(40015)  # UIA IsReadOnlyAttribute
            return value is False or (type(value) is int and value == 0)
        except Exception:
            return False

    @staticmethod
    def _read_control_text(node, limit: int = 4096) -> str:
        """Read editor contents via UIA patterns, independently of the control name."""
        try:
            if node.IsPassword:
                return ""
            if node.ControlTypeName not in {"EditControl", "DocumentControl", "ComboBoxControl"}:
                return ""
        except Exception:
            return ""
        try:
            pattern = node.GetTextPattern()
            if pattern is not None:
                text = pattern.DocumentRange.GetText(limit + 1)
                if isinstance(text, str) and text:
                      truncated = len(text) > limit
                      text = text.replace('\r\n', '\n').replace('\r', '\n')
                      return text[:limit] + ("\n[content truncated]" if truncated else "")
        except Exception:
            pass
        try:
            pattern = node.GetValuePattern()
            text = pattern.Value if pattern is not None else ""
            if isinstance(text, str):
                  truncated = len(text) > limit
                  text = text.replace('\r\n', '\n').replace('\r', '\n')
                  return text[:limit] + ("\n[content truncated]" if truncated else "")
        except Exception:
            pass
        return ""

    def _walk_tree(
        self,
        node,
        interactive: list[UIElement],
        context: list[UIElement],
        max_depth: int,
        depth: int = 0,
        max_nodes: int = 512,
    ):
        # File lists can contain hundreds of editable metadata cells. Keep
        # their rows as secondary candidates so they cannot hide dialog fields
        # (File name, file type, Save/Cancel) later in the accessibility tree.
        pending = deque([(node, depth, False, False)])
        document_controls: list[UIElement] = []
        rows: list[UIElement] = []
        visited = 0
        stats = {'visited': 0, 'depth_cutoffs': 0, 'child_cutoffs': 0,
                 'node_limit': False, 'errors': 0, 'deep_retry': False}
        while pending and visited < max_nodes:
            current, level, in_row, in_document = pending.popleft()
            visited += 1
            if level >= max_depth:
                stats['depth_cutoffs'] += 1
                continue
            try:
                ctrl_type = current.ControlTypeName
                in_row = in_row or ctrl_type in {"ListItemControl", "TreeItemControl"}
                if ctrl_type in ALL_CAPTURED_CONTROL_TYPES:
                    rect = current.BoundingRectangle
                    focus_only = False
                    runtime_id = ()
                    if (ctrl_type == 'EditControl' and rect and (rect.width() <= 0 or rect.height() <= 0)
                            and current.IsEnabled and not current.IsPassword
                            and not getattr(current, 'IsOffscreen', False)):
                        runtime_id = tuple(current.GetRuntimeId())
                        focus_only = bool(runtime_id)
                    if (focus_only or (rect and rect.width() > 0 and rect.height() > 0
                            and rect.left >= 0 and rect.top >= 0
                            and not getattr(current, "IsOffscreen", False))):
                        editable_document = self._is_editable_document(current)
                        is_interactive = ctrl_type in INTERACTIVE_CONTROL_TYPES or editable_document
                        bucket = (rows if in_row else document_controls if in_document else interactive) if is_interactive else context
                        limit = self.max_elements if is_interactive else self.context_budget
                        editable_control = editable_document or ctrl_type in ('EditControl', 'ComboBoxControl')
                        if editable_control and not runtime_id:
                            try:
                                runtime_id = tuple(current.GetRuntimeId())
                            except Exception:
                                runtime_id = ()
                        prioritize = editable_control and not in_row
                        if len(bucket) < limit or prioritize:
                            element = UIElement(
                                tag=0, role=ctrl_type,
                                name=(current.Name or "").strip()[:80],
                                bounds=(rect.left, rect.top, rect.right, rect.bottom),
                                is_enabled=current.IsEnabled,
                                automation_id=current.AutomationId or "",
                                is_interactive=is_interactive,
                                text=self._read_control_text(current),
                                is_editable=editable_control,
                                runtime_id=runtime_id, focus_only=focus_only,
                            )
                            if prioritize and len(bucket) >= limit:
                                replace = next((i for i in range(len(bucket) - 1, -1, -1)
                                                if not bucket[i].is_editable), None)
                                if replace is not None:
                                    bucket[replace] = element
                            elif len(bucket) < limit:
                                bucket.append(element)
            except Exception:
                # A broken control must not suppress its siblings.
                stats['errors'] += 1
            try:
                child_limit = max(32, self.max_elements)
                children = list(islice(current.GetChildren(), child_limit + 1))
                if len(children) > child_limit:
                    stats['child_cutoffs'] += 1
                for child in children[:child_limit]:
                    if len(pending) + visited >= max_nodes:
                        stats['node_limit'] = True
                        break
                    pending.append((child, level + 1, in_row,
                                    in_document or current.ControlTypeName == 'DocumentControl'))
            except Exception:
                stats['errors'] += 1
        if document_controls:
            # Reserve room for native navigation controls while preventing a
            # browser toolbar from hiding all actionable document descendants.
            page_budget = max(1, self.max_elements * 3 // 4)
            document_controls.sort(key=lambda e: not e.is_editable)
            selected = document_controls[:page_budget]
            native_budget = self.max_elements - len(selected)
            selected += sorted(interactive, key=lambda e: not e.is_editable)[:native_budget]
            selected += document_controls[page_budget:page_budget + max(0, self.max_elements-len(selected))]
            interactive[:] = selected
        interactive.extend(rows[:max(0, self.max_elements - len(interactive))])
        stats['visited'] = visited
        stats['node_limit'] = stats['node_limit'] or bool(pending)
        self._tree_stats = stats

    def accessibility_diagnostic(self):
        stats = getattr(self, '_tree_stats', {})
        if not stats:
            return 'Accessibility scan unavailable.'
        return ('UIA scan: ' + ', '.join(f'{key}={value}' for key, value in stats.items()))

    def annotate_screenshot(self, img: Image.Image, elements: list[UIElement]) -> Image.Image:
        """Draw red numbered tags + bounding boxes on each element."""
        annotated = img.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
                )
            except Exception:
                font = ImageFont.load_default()

        for el in elements:
            if el.focus_only:
                continue
            l, t, r, b = el.bounds
            color = (255, 0, 0) if el.is_enabled else (128, 128, 128)
            draw.rectangle([l, t, r, b], outline=color, width=2)
            tag_text = str(el.tag)
            tx, ty = l, max(t - 22, 0)
            draw.rectangle([tx, ty, tx + max(28, len(tag_text) * 12 + 8), ty + 22], fill=color)
            draw.text((tx + 4, ty), tag_text, fill=(255, 255, 255), font=font)
        return annotated

    def annotate_with_target(
        self,
        img: Image.Image,
        elements: list[UIElement],
        target_tag: Optional[int] = None,
    ) -> Image.Image:
        """Same as annotate_screenshot, but additionally highlights the target tag with a thick blue ring."""
        annotated = self.annotate_screenshot(img, elements)
        if target_tag is None:
            return annotated
        target = next((e for e in elements if e.tag == target_tag), None)
        if target is None:
            return annotated

        draw = ImageDraw.Draw(annotated)
        l, t, r, b = target.bounds
        # Thick blue ring
        for offset in range(0, 6, 2):
            draw.rectangle(
                [l - offset, t - offset, r + offset, b + offset],
                outline=(70, 140, 240), width=2,
            )
        return annotated

    def _resize_for_api(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        long_edge = max(w, h)
        if long_edge <= self.max_dimension:
            return img
        scale = self.max_dimension / long_edge
        return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    def img_to_b64(self, img: Image.Image) -> str:
        img = self._resize_for_api(img)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def perceive(self) -> ScreenState:
        """One-shot: capture everything the reasoning core needs."""
        elements, window_title = self.get_accessibility_tree()
        # Common file dialogs can expose only their title-bar Close control
        # during creation. Retry observation once, without model calls/input.
        if (window_title.strip().casefold() in {'open', 'save as'} and
                all(e.role == 'ButtonControl' and e.name.casefold() == 'close'
                    for e in elements)):
            time.sleep(0.3)
            elements, window_title = self.get_accessibility_tree()
        # The tree scan can take time. Capture the evidence image after it so
        # transient UI changes are not paired with an earlier screenshot.
        raw_img = self.capture_screen()
        if contains_credentials('\n'.join([window_title] + [f'{e.name}\n{e.text}' for e in elements])):
            # Do not retain credential-bearing editor text, titles or images in
            # model input, dashboard previews, or task-history observations.
            return ScreenState('', '', [], '[Sensitive content hidden]', raw_img.width, raw_img.height,
                               getattr(self, '_foreground_handle', 0),
                               getattr(self, '_foreground_process_id', 0), True)
        annotated = self.annotate_screenshot(raw_img, elements)
        field_images = self.field_crops(raw_img, elements)
        return ScreenState(
            screenshot_b64=self.img_to_b64(annotated),
            raw_screenshot_b64=self.img_to_b64(raw_img),
            elements=elements,
            foreground_window_title=window_title,
            width=raw_img.width,
            height=raw_img.height,
            foreground_window_handle=getattr(self, '_foreground_handle', 0),
            foreground_process_id=getattr(self, '_foreground_process_id', 0),
            accessibility_diagnostic=self.accessibility_diagnostic(),
            field_images=field_images,
        )

    @staticmethod
    def field_crops(img, elements):
        """Bounded unannotated field evidence at original screen resolution."""
        crops = {}
        for element in elements:
            if len(crops) >= 12:
                break
            if not element.is_enabled or element.focus_only or not (element.is_editable or element.role in {'EditControl', 'ComboBoxControl'}):
                continue
            left, top, right, bottom = element.bounds
            if right <= left or bottom <= top or left < 0 or top < 0 or right > img.width or bottom > img.height:
                continue
            # Include a small border for the field edge without tagging over text.
            crop = img.crop((max(0, left-12), max(0, top-12), min(img.width, right+12), min(img.height, bottom+12)))
            if crop.width > 1600 or crop.height > 600:
                continue
            buffer = io.BytesIO()
            crop.save(buffer, format='PNG')
            crops[element.tag] = base64.b64encode(buffer.getvalue()).decode('ascii')
        return crops


if __name__ == "__main__":
    engine = VisualPerceptionEngine()
    state = engine.perceive()
    print(f"Foreground window: {state.foreground_window_title}")
    print(f"Found {len(state.elements)} interactive elements")
    print(f"Screen: {state.width}x{state.height}")
    for el in state.elements[:10]:
        print(f"  [{el.tag}] {el.role}: '{el.name}' @ {el.bounds}")
    annotated = engine.annotate_screenshot(engine.capture_screen(), state.elements)
    annotated.save("perception_debug.png")
    print("Saved perception_debug.png — open this to see what the LLM sees.")

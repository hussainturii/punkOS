"""
M7: Web Interaction Specialist — Playwright Agent (SDS 1.7.7, FR5.1-FR5.7).

Why a separate agent instead of driving the browser through the accessibility
tree like every other app: Chrome and Electron expose a shallow or empty UIA
tree, which is exactly the M1 blind spot named in the SDS limitations. The DOM
is the browser's own ground truth — richer, stable, and queryable — so web work
is routed here instead (FE-1, and OB-4's 90% latency reduction).

Covers:
  FR5.1  live page preview (screenshot per action)
  FR5.2  multi-tab coordination, hard-capped at 5 (business rule)
  FR5.3  DOM tree snapshot for the explorer panel
  FR5.4  per-field form fill status tracking
  FR5.5  generated Playwright script, recorded for user review
  FR5.6  contextual form filling by label / placeholder / aria / name
  FR5.7  structured scraping to JSON or CSV
  SEC-8  every selector is regex-validated AND existence-checked before use

Playwright is imported lazily so the rest of PunkOS runs on machines that never
installed it. Install with:  pip install playwright && playwright install chromium
"""
from __future__ import annotations

import base64
import csv
import io
import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# SEC-8: selector whitelist
# ---------------------------------------------------------------------------

# LLM-generated selectors are untrusted input. We allow ordinary CSS (tags,
# #ids, .classes, [attr=value], combinators, :nth-of-type) and Playwright's
# text=/role= engines, and reject anything carrying script-injection shapes
# such as javascript: URLs, angle brackets, or statement separators.
_SELECTOR_ALLOWED = re.compile(
    r"^[A-Za-z0-9_\-\#\.\[\]\=\"'\:\(\)\s\>\+\~\*\,\/\$\^\|@]+$"
)
_SELECTOR_FORBIDDEN = re.compile(
    r"(javascript\s*:|<\s*script|</|\bon\w+\s*=|;\s*\w+\s*\(|\beval\b|\bdocument\.)",
    re.IGNORECASE,
)


def validate_selector(selector: str) -> tuple[bool, str]:
    """SEC-8. Returns (is_valid, reason)."""
    if not selector or not selector.strip():
        return False, "Selector is empty"
    if len(selector) > 500:
        return False, "Selector is unreasonably long (possible injection)"
    if _SELECTOR_FORBIDDEN.search(selector):
        return False, "Selector contains a script-injection pattern"
    if not _SELECTOR_ALLOWED.match(selector):
        return False, "Selector contains characters outside the allowed set"
    return True, "OK"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class FillStatus(Enum):
    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    FAILED = "Failed"


@dataclass
class FormField:
    """One identified form field and its fill state (FR5.4)."""
    selector: str
    label: str
    field_type: str
    placeholder: str = ""
    name: str = ""
    required: bool = False
    status: FillStatus = FillStatus.PENDING
    value: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "selector": self.selector, "label": self.label,
            "type": self.field_type, "placeholder": self.placeholder,
            "name": self.name, "required": self.required,
            "status": self.status.value, "value": self.value, "error": self.error,
        }


@dataclass
class DomNode:
    """One node of the DOM tree explorer (FR5.3)."""
    tag: str
    node_id: str
    classes: str
    text: str
    selector: str
    children: list["DomNode"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tag": self.tag, "id": self.node_id, "classes": self.classes,
            "text": self.text, "selector": self.selector,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class TabInfo:
    index: int
    title: str
    url: str
    status: str = "idle"


@dataclass
class WebResult:
    success: bool
    message: str
    data: Any = None
    screenshot_b64: str = ""


# JS that walks the live DOM and returns a compact tree. Runs inside the page.
_DOM_SNAPSHOT_JS = """
(maxDepth) => {
  function selectorFor(el) {
    if (el.id) return '#' + CSS.escape(el.id);
    let path = el.tagName.toLowerCase();
    if (el.className && typeof el.className === 'string') {
      const cls = el.className.trim().split(/\\s+/).filter(Boolean).slice(0, 2);
      if (cls.length) path += '.' + cls.map(c => CSS.escape(c)).join('.');
    }
    return path;
  }
  function walk(el, depth) {
    if (depth > maxDepth) return null;
    const node = {
      tag: el.tagName.toLowerCase(),
      id: el.id || '',
      classes: (typeof el.className === 'string' ? el.className : ''),
      text: (el.childElementCount === 0 ? (el.textContent || '').trim().slice(0, 80) : ''),
      selector: selectorFor(el),
      children: []
    };
    for (const child of el.children) {
      if (['script','style','noscript','svg'].includes(child.tagName.toLowerCase())) continue;
      const c = walk(child, depth + 1);
      if (c) node.children.push(c);
      if (node.children.length >= 40) break;
    }
    return node;
  }
  return walk(document.body, 0);
}
"""

# JS that identifies every fillable field plus the context needed to match it
# to a semantic label (FR5.6).
_FORM_FIELDS_JS = """
() => {
  const out = [];
  const els = document.querySelectorAll('input, textarea, select');
  els.forEach((el, i) => {
    const type = (el.getAttribute('type') || el.tagName).toLowerCase();
    if (['hidden','submit','button','reset','image'].includes(type)) return;

    let label = '';
    if (el.id) {
      const lab = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
      if (lab) label = lab.textContent.trim();
    }
    if (!label && el.closest('label')) label = el.closest('label').textContent.trim();
    if (!label) label = el.getAttribute('aria-label') || '';
    if (!label) label = el.getAttribute('placeholder') || '';
    if (!label) label = el.getAttribute('name') || '';

    let selector;
    if (el.id) selector = '#' + CSS.escape(el.id);
    else if (el.getAttribute('name')) selector = el.tagName.toLowerCase() + '[name="' + el.getAttribute('name') + '"]';
    else selector = el.tagName.toLowerCase() + ':nth-of-type(' + (i + 1) + ')';

    out.push({
      selector: selector,
      label: label.slice(0, 100),
      type: type,
      placeholder: el.getAttribute('placeholder') || '',
      name: el.getAttribute('name') || '',
      required: el.hasAttribute('required')
    });
  });
  return out;
}
"""


class WebAgent:
    """Playwright-backed browser controller."""

    MAX_TABS = 5   # FR5.2 business rule

    def __init__(
        self,
        browser: str = "chromium",
        headless: bool = False,
        max_tabs: int = 5,
        navigation_timeout_ms: int = 30000,
        on_log: Optional[Callable[[str], None]] = None,
        audit_log=None,
        channel: Optional[str] = None,
        download_dir: Optional[str] = None,
    ):
        self.browser_name = browser
        self.headless = headless
        self.max_tabs = min(max_tabs, self.MAX_TABS)
        self.navigation_timeout_ms = navigation_timeout_ms
        self.on_log = on_log or (lambda _m: None)
        self.audit_log = audit_log
        self.channel = channel
        self.download_dir = Path(download_dir or Path.home() / 'Downloads' / 'PunkOS')
        self.downloads = []

        self._playwright = None
        self._browser = None
        self._context = None
        self._pages: list = []
        self._active_index = 0
        self._navigation_receipts = {}
        self._navigation_sequence = 0
        self._pending_popups = []
        self._popup_error = ''
        self._task_tabs = {}
        self._unassigned_navigation = []
        # FR5.5: every action appends the equivalent Playwright call here so the
        # user can read exactly what was run.
        self.script_lines: list[str] = []
        self.form_fields: list[FormField] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._browser is not None and self._browser.is_connected()

    def start(self) -> WebResult:
        if self.is_running:
            if self.page is None or self.page.is_closed():
                return WebResult(False, 'The active tab is closed. Open a new tab or switch to an existing tab.')
            return WebResult(True, "Browser already running")
        if self._playwright is not None:
            # A browser can be closed outside PunkOS. Release its old driver
            # before starting another sync Playwright instance.
            self.close()
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return WebResult(
                False,
                "Playwright is not installed. Run:\n"
                "  pip install playwright\n"
                "  playwright install chromium",
            )
        try:
            self._playwright = sync_playwright().start()
            launcher = getattr(self._playwright, self.browser_name, self._playwright.chromium)
            channels = [self.channel] if self.channel else ([None, 'chrome', 'msedge'] if self.browser_name == 'chromium' else [None])
            errors = []
            for channel in channels:
                try:
                    self._browser = launcher.launch(headless=self.headless, **({'channel': channel} if channel else {}))
                    self.actual_channel = channel or self.browser_name
                    self.on_log(f'[web] launched {channel or self.browser_name}')
                    break
                except Exception as exc:
                    errors.append(str(exc).splitlines()[0])
            if self._browser is None:
                raise RuntimeError('; '.join(errors) + f'\nInstall the matching browser with: "{sys.executable}" -m playwright install {self.browser_name}')
            self._context = self._browser.new_context(accept_downloads=True)
            self._context.set_default_timeout(self.navigation_timeout_ms)
            self._context.on('page', self._register_page)
            self._context.on('response', self._context_response)
            page = self._context.new_page()
            self._register_page(page)
            self._pages = [page]
            self._active_index = 0
            self.script_lines = [
                "from playwright.sync_api import sync_playwright",
                "p = sync_playwright().start()",
                f"browser = p.{self.browser_name}.launch(headless={self.headless}, **{({'channel': channel} if channel else {})!r})",
                "context = browser.new_context()",
                "page = context.new_page()",
            ]
            self.on_log(f"[web] {self.browser_name} launched (headless={self.headless})")
            return WebResult(True, f"{self.actual_channel} launched")
        except Exception as e:
            self.close()
            return WebResult(False, f"Failed to launch browser: {e}")

    def close(self) -> WebResult:
        """Tear everything down. Safe to call repeatedly."""
        errors = []
        for closer in (
            lambda: self._context and self._context.close(),
            lambda: self._browser and self._browser.close(),
            lambda: self._playwright and self._playwright.stop(),
        ):
            try:
                closer()
            except Exception as e:
                errors.append(str(e))
        self._context = self._browser = self._playwright = None
        self._pages = []
        self._active_index = 0
        self._pending_popups.clear()
        self._popup_error = ''
        self._task_tabs.clear()
        self._navigation_receipts.clear()
        self._unassigned_navigation.clear()
        self.on_log("[web] browser closed")
        return WebResult(not errors, "Browser closed" if not errors else f"Closed with errors: {errors}")

    @property
    def page(self):
        if not self._pages:
            return None
        return self._pages[self._active_index]

    def _require_page(self) -> Optional[WebResult]:
        if not self.is_running or self.page is None:
            return WebResult(False, "Browser is not running — call start() first")
        if self.page.is_closed():
            return WebResult(False, 'The active tab is closed. Open a new tab or switch to an existing tab.')
        return None

    # ------------------------------------------------------------------
    # Navigation (FR5.1)
    # ------------------------------------------------------------------

    def navigate(self, url: str) -> WebResult:
        err = self._require_page()
        if err:
            return err
        url = self.normalize_url(url)
        try:
            response = self.page.goto(url, wait_until="domcontentloaded")
            self.script_lines.append(f"page.goto({url!r})")
            if response is not None and response.status >= 400:
                message = (f'Navigation failed: HTTP {response.status} at {self.page.url}. '
                           'Return to the last working listing and follow an observed link, '
                           'or search for the requested page. Do not guess another URL or '
                           'treat the error page as release evidence.')
                self.on_log('[web] ' + message)
                return WebResult(False, message, data={'url': self.page.url, 'status': response.status})
            if self.audit_log is not None:
                self.audit_log.record("web_navigate", f"navigated to {url}", {"url": url})
            self.on_log(f"[web] -> {url}")
            return WebResult(True, f"Navigated to {url}", data={"url": self.page.url,
                                                                "title": self.page.title()})
        except Exception as e:
            return WebResult(False, f"Navigation failed: {e}")

    def _track_navigation(self, page):
        """Record main-document responses only, including click navigations."""
        def response_seen(response):
            try:
                request = response.request
                if not request.is_navigation_request() or request.frame != page.main_frame:
                    return
                if 300 <= response.status < 400:
                    return  # The redirect response is not the destination page.
                self._navigation_sequence += 1
                self._navigation_receipts[id(page)] = {
                    'sequence': self._navigation_sequence,
                    'url': response.url, 'status': response.status}
            except Exception as exc:
                self.on_log(f'[web] Navigation response unavailable ({type(exc).__name__})')
        page.on('response', response_seen)

    def _context_response(self, response):
        # A popup's initial response may precede its page event/listener.
        try:
            request = response.request
            if not request.is_navigation_request():
                return
            try:
                page = request.frame.page
            except Exception:
                # Chromium can expose an initial popup navigation before its
                # frame exists. Keep only navigation responses, not assets.
                if not 300 <= response.status < 400:
                    self._navigation_sequence += 1
                    self._unassigned_navigation.append({'sequence': self._navigation_sequence,
                        'url': response.url, 'status': response.status})
                    self._unassigned_navigation = self._unassigned_navigation[-16:]
                return
            if request.frame != page.main_frame or 300 <= response.status < 400:
                return
            self._navigation_sequence += 1
            self._navigation_receipts[id(page)] = {
                'sequence': self._navigation_sequence, 'url': response.url, 'status': response.status}
        except Exception as exc:
            self.on_log(f'[web] Context navigation response unavailable ({type(exc).__name__})')

    def _register_page(self, page):
        if page in self._pages:
            return
        if len(self._pages) >= self.max_tabs:
            self._popup_error = 'Browser tab limit exceeded by a new page. Close an unused controlled tab before retrying.'
            page.close()
            return
        self._track_navigation(page)
        self._pages.append(page)
        if page.opener() is not None:
            self._pending_popups.append(page)

    def _adopt_popup(self):
        pending = [p for p in self._pending_popups if not p.is_closed()]
        self._pending_popups.clear()
        if self._popup_error:
            return self._popup_error
        if len(pending) == 1:
            try:
                pending[0].wait_for_load_state('domcontentloaded')
                candidates = [r for r in self._unassigned_navigation if r['url'] == pending[0].url]
                if len(candidates) == 1 and id(pending[0]) not in self._navigation_receipts:
                    self._navigation_receipts[id(pending[0])] = candidates[0]
                    self._unassigned_navigation.remove(candidates[0])
                pending[0].bring_to_front()
                self._active_index = self._pages.index(pending[0])
                self._task_targets = {}
            except Exception as exc:
                return f'Popup could not be observed: {type(exc).__name__}: {exc}'
        elif len(pending) > 1:
            return 'Multiple popup pages opened; target page is ambiguous. Select a tab explicitly before retrying.'
        return ''

    def _current_navigation(self):
        receipt = self._navigation_receipts.get(id(self.page), {})
        # A response can arrive before the document commits. Never associate
        # an old/new document's response with a different visible page.
        if receipt and receipt['url'].split('#', 1)[0] == self.page.url.split('#', 1)[0]:
            return dict(receipt)
        return {}

    @staticmethod
    def normalize_url(url: str) -> str:
        """Add a scheme to a bare hostname, and leave every other scheme alone.

        Blindly prefixing `https://` turned `file:///C:/page.html` into
        `https://file:///C:/page.html`, so local pages could not be opened at
        all. Anything already carrying a scheme is passed through untouched.
        """
        url = (url or "").strip()
        if not url:
            return url
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:", url):
            return url
        return "https://" + url.lstrip("/")

    def screenshot_b64(self) -> str:
        """FR5.1: page preview for the UI."""
        if not self.is_running or self.page is None:
            return ""
        try:
            return base64.b64encode(self.page.screenshot(type="png")).decode("ascii")
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Multi-tab coordination (FR5.2)
    # ------------------------------------------------------------------

    def new_tab(self, url: Optional[str] = None) -> WebResult:
        if not self.is_running:
            return WebResult(False, "Browser is not running — call start() first")
        # Retire externally closed pages when explicitly creating a new tab.
        active = self.page
        self._pages = [page for page in self._pages if not page.is_closed()]
        self._active_index = self._pages.index(active) if active in self._pages else 0
        if len(self._pages) >= self.max_tabs:
            return WebResult(False, f"Tab limit reached ({self.max_tabs} concurrent tabs max)")
        try:
            page = self._context.new_page()
            self._register_page(page)
            self._active_index = len(self._pages) - 1
            self.script_lines.append("page = context.new_page()")
            if url:
                return self.navigate(url)
            return WebResult(True, f"Opened tab {self._active_index + 1}")
        except Exception as e:
            return WebResult(False, f"Failed to open tab: {e}")

    def switch_tab(self, index: int) -> WebResult:
        if not 0 <= index < len(self._pages):
            return WebResult(False, f"No tab at index {index}")
        try:
            self._pages[index].bring_to_front()
        except Exception as exc:
            return WebResult(False, f"Failed to switch to tab {index + 1}: {exc}")
        self._active_index = index
        return WebResult(True, f"Switched to tab {index + 1}")

    def close_tab(self, index: int) -> WebResult:
        if not 0 <= index < len(self._pages):
            return WebResult(False, f"No tab at index {index}")
        if len(self._pages) == 1:
            return WebResult(False, "Cannot close the last remaining tab")
        try:
            self._pages[index].close()
        except Exception as exc:
            return WebResult(False, f"Failed to close tab {index + 1}: {exc}")
        self._pages.pop(index)
        self._popup_error = ''
        if index < self._active_index:
            self._active_index -= 1
        else:
            self._active_index = min(self._active_index, len(self._pages) - 1)
        return WebResult(True, f"Closed tab {index + 1}")

    def list_tabs(self) -> list[TabInfo]:
        tabs = []
        for i, p in enumerate(self._pages):
            try:
                tabs.append(TabInfo(i, p.title() or "(untitled)", p.url,
                                    "active" if i == self._active_index else "idle"))
            except Exception:
                tabs.append(TabInfo(i, "(unavailable)", "", "error"))
        return tabs

    # ------------------------------------------------------------------
    # DOM inspection (FR5.3)
    # ------------------------------------------------------------------

    def dom_snapshot(self, max_depth: int = 6) -> Optional[DomNode]:
        if not self.is_running or self.page is None:
            return None
        try:
            raw = self.page.evaluate(_DOM_SNAPSHOT_JS, max_depth)
            return self._to_dom_node(raw) if raw else None
        except Exception as e:
            self.on_log(f"[web] DOM snapshot failed: {e}")
            return None

    def _to_dom_node(self, raw: dict) -> DomNode:
        return DomNode(
            tag=raw.get("tag", ""), node_id=raw.get("id", ""),
            classes=raw.get("classes", ""), text=raw.get("text", ""),
            selector=raw.get("selector", ""),
            children=[self._to_dom_node(c) for c in raw.get("children", [])],
        )

    # ------------------------------------------------------------------
    # Interaction — every selector passes SEC-8 then an existence check
    # ------------------------------------------------------------------

    def _check_selector(self, selector: str) -> Optional[WebResult]:
        ok, reason = validate_selector(selector)
        if not ok:
            if self.audit_log is not None:
                self.audit_log.record("web_selector_rejected",
                                      f"{reason}: {selector[:120]}",
                                      {"selector": selector})
            return WebResult(False, f"Selector rejected ({reason}): {selector[:80]}")
        try:
            if self.page.query_selector(selector) is None:
                return WebResult(False, f"No element matches selector: {selector}")
        except Exception as e:
            return WebResult(False, f"Selector could not be evaluated: {e}")
        return None

    def click(self, selector: str) -> WebResult:
        err = self._require_page() or self._check_selector(selector)
        if err:
            return err
        try:
            self.page.click(selector)
            self.script_lines.append(f"page.click({selector!r})")
            return WebResult(True, f"Clicked {selector}")
        except Exception as e:
            return WebResult(False, f"Click failed on {selector}: {e}")

    def task_observation(self, _popup_retry=False):
        """Bounded visible DOM, with exact selectors emitted by the page itself."""
        popup_error = self._adopt_popup()
        if popup_error:
            return {'error': popup_error}
        if self._require_page():
            return {'error': 'Browser unavailable'}
        rows = self.page.locator('a, button, input, textarea, select, [role="button"]').evaluate_all('''els => els.map((e, i) => {
            const r = e.getBoundingClientRect();
            const style = getComputedStyle(e);
            return {index:i, tag:e.tagName, text:(e.innerText || e.getAttribute('aria-label') || e.placeholder || '').slice(0,120),
              href:e.href || '', type:e.type || '', download:e.hasAttribute('download'), opens_new_tab:e.target==='_blank', visible:r.width>0 && r.height>0 && style.visibility!=='hidden', disabled:!!e.disabled};
        }).filter(e=>e.visible).slice(0,60)''')
        self._task_targets = {str(r['index']): r for r in rows}
        # Playwright may dispatch popup events while evaluating the source DOM.
        # Refresh once on the adopted page rather than hand the model stale tags.
        if self._pending_popups:
            if _popup_retry:
                return {'error': 'New popups kept appearing during observation; task stopped.'}
            return self.task_observation(_popup_retry=True)
        self._task_tabs = {str(i): page for i, page in enumerate(self._pages) if not page.is_closed()}
        return {'url': self.page.url, 'title': self.page.title(),
                'text': self.page.locator('body').inner_text()[:6000], 'targets': rows,
                'downloads': self.downloads, 'navigation': self._current_navigation(),
                'active_tab': self._active_index,
                'tabs': [{'index': key, 'url': page.url, 'title': page.title()} for key,page in self._task_tabs.items()]}

    def task_action(self, action, target='', text=''):
        from urllib.parse import quote_plus, urlsplit
        if action == 'switch_tab':
            page = self._task_tabs.get(str(target))
            if page is None or page not in self._pages or page.is_closed():
                return WebResult(False, 'Tab is absent from the current observation.')
            result = self.switch_tab(self._pages.index(page))
            if result.success:
                self._task_targets = {}
            return result
        if action in ('navigate', 'search'):
            url = 'https://www.google.com/search?q=' + quote_plus(text) if action == 'search' else text.strip()
            try:
                parsed = urlsplit(url)
                valid = (parsed.scheme in ('http', 'https') and bool(parsed.hostname)
                         and not any(c.isspace() for c in url)
                         and not parsed.username and not parsed.password)
                parsed.port  # Reject malformed ports before calling the browser.
            except ValueError:
                valid = False
            if not valid:
                return WebResult(False, 'Invalid navigation URL. Put only the complete http:// or https:// URL in text, without prose. target must be empty.')
            return self.navigate(url)
        if action == 'wait':
            self.page.wait_for_timeout(700)
            return WebResult(True, 'Waited for page update')
        row = getattr(self, '_task_targets', {}).get(str(target))
        if row is None:
            return WebResult(False, 'Target is absent from the current observation.')
        locator = self.page.locator('a, button, input, textarea, select, [role="button"]').nth(row['index'])
        try:
            # Reject targets whose identity changed during inference.
            current = locator.evaluate('(e)=>({tag:e.tagName, href:e.href || "", download:e.hasAttribute("download"), text:(e.innerText || e.getAttribute("aria-label") || e.placeholder || "").slice(0,120)})')
            if any(current[k] != row[k] for k in ('tag', 'href', 'text', 'download')):
                return WebResult(False, 'Page changed; observe and select the target again.')
            # A native download link must persist its file for either action.
            if action == 'click' and row['download']:
                action = 'download'
            if action == 'click':
                if row.get('opens_new_tab'):
                    with self.page.expect_popup() as pending_popup:
                        locator.click()
                    self._register_page(pending_popup.value)
                else:
                    locator.click()
            elif action == 'fill':
                locator.fill(text)
            elif action == 'press':
                if text not in ('Enter', 'Tab', 'Escape', 'ArrowDown', 'ArrowUp'):
                    return WebResult(False, 'Unsupported web key')
                locator.press(text)
            elif action == 'download':
                with self.page.expect_download() as pending:
                    locator.click()
                download = pending.value
                self.download_dir.mkdir(parents=True, exist_ok=True)
                import uuid
                filename = re.sub(r'[^A-Za-z0-9._-]', '_', download.suggested_filename)[:160] or 'download'
                destination = self.download_dir / (uuid.uuid4().hex[:10] + '-' + filename)
                download.save_as(str(destination))
                if download.failure():
                    return WebResult(False, 'Download failed: ' + str(download.failure()))
                receipt = {'path': str(destination), 'bytes': destination.stat().st_size, 'url': download.url}
                self.downloads.append(receipt)
                return WebResult(True, 'Download persisted', receipt)
            else:
                return WebResult(False, 'Unsupported web action')
            return WebResult(True, f'{action} executed on {target}')
        except Exception as exc:
            return WebResult(False, f'{action} failed: {type(exc).__name__}: {str(exc)[:250]}')

    def fill(self, selector: str, value: str) -> WebResult:
        err = self._require_page() or self._check_selector(selector)
        if err:
            return err
        try:
            self.page.fill(selector, value)
            self.script_lines.append(f"page.fill({selector!r}, {value!r})")
            return WebResult(True, f"Filled {selector}")
        except Exception as e:
            return WebResult(False, f"Fill failed on {selector}: {e}")

    # ------------------------------------------------------------------
    # Intelligent form filling (FR5.4 + FR5.6)
    # ------------------------------------------------------------------

    def current_page_label(self) -> str:
        """A short 'Title — url' for status messages, so the user can see which
        page an operation actually looked at."""
        if not self.is_running or self.page is None:
            return "no page"
        try:
            title = (self.page.title() or "").strip()
            url = self.page.url
            return f"{title} — {url}" if title else url
        except Exception:
            return "no page"

    def discover_form_fields(self) -> list[FormField]:
        """Identify every fillable field with its semantic context.

        Raises nothing: a page with no form is a normal outcome, not an error.
        Callers should use `last_discovery_error` to tell "this page genuinely
        has no fields" apart from "the scan itself broke".
        """
        self.last_discovery_error: Optional[str] = None
        if not self.is_running or self.page is None:
            self.last_discovery_error = "Browser is not running"
            return []
        try:
            raw = self.page.evaluate(_FORM_FIELDS_JS)
        except Exception as e:
            self.last_discovery_error = str(e)
            self.on_log(f"[web] form discovery failed: {e}")
            return []

        self.form_fields = [
            FormField(
                selector=f["selector"], label=f["label"], field_type=f["type"],
                placeholder=f["placeholder"], name=f["name"], required=f["required"],
            )
            for f in raw
        ]
        return self.form_fields

    @staticmethod
    def _normalize(text: str) -> str:
        """Fold separators so 'full_name_x9f2' and 'Full Name' compare equal.

        Real forms name fields full_name, full-name, fullName or full_name_x9f2
        (framework-generated suffixes). Normalising camelCase and separator
        characters to spaces is what makes FR5.6 work on fields whose ids are
        'complex, missing, or dynamically generated'.
        """
        spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)     # camelCase -> camel Case
        spaced = re.sub(r"[_\-\.\[\]]+", " ", spaced)
        return re.sub(r"\s+", " ", spaced).strip().lower()

    @staticmethod
    def match_field(field: FormField, values: dict[str, str]) -> Optional[str]:
        """FR5.6: map a user-supplied key onto a discovered field.

        Matching runs over label, placeholder, name and type together, so a
        field with no id and a generated name still resolves as long as *some*
        piece of its context mentions the concept. Longest key first, so
        'confirm password' beats a bare 'password'.
        """
        haystack = WebAgent._normalize(" ".join([
            field.label, field.placeholder, field.name, field.field_type
        ]))
        for key in sorted(values, key=len, reverse=True):
            norm_key = WebAgent._normalize(key)
            if not norm_key:
                continue
            # Word-boundary match so 'name' doesn't fire on 'username'.
            if re.search(rf"\b{re.escape(norm_key)}\b", haystack):
                return values[key]
        return None

    def fill_form(
        self,
        values: dict[str, str],
        on_progress: Optional[Callable[[FormField], None]] = None,
    ) -> WebResult:
        """Fill every field we can match, tracking per-field status (FR5.4)."""
        err = self._require_page()
        if err:
            return err

        fields = self.discover_form_fields()
        if not fields:
            error = getattr(self, "last_discovery_error", None)
            if error:
                return WebResult(False, f"Could not scan the page for fields: {error}")
            return WebResult(
                False,
                f"This page has no form fields to fill ({self.current_page_label()}). "
                f"Navigate to a page with a form first."
            )

        filled = 0
        for fld in fields:
            value = self.match_field(fld, values)
            if value is None:
                continue

            fld.status = FillStatus.IN_PROGRESS
            if on_progress:
                on_progress(fld)

            result = self.fill(fld.selector, value)
            if result.success:
                fld.status = FillStatus.COMPLETED
                fld.value = value
                filled += 1
            else:
                fld.status = FillStatus.FAILED
                fld.error = result.message
            if on_progress:
                on_progress(fld)

        unmatched = [f.label or f.selector for f in fields
                     if f.required and f.status != FillStatus.COMPLETED]
        message = f"Filled {filled} of {len(fields)} fields"
        if unmatched:
            message += f"; required fields still empty: {', '.join(unmatched)}"
        return WebResult(bool(filled), message, data=[f.to_dict() for f in fields])

    def verify_fill(self) -> WebResult:
        """FR5.6 business rule: post-fill DOM comparison before submission."""
        mismatches = []
        for fld in self.form_fields:
            if fld.status != FillStatus.COMPLETED:
                continue
            try:
                actual = self.page.input_value(fld.selector)
                if actual != fld.value:
                    mismatches.append(f"{fld.label or fld.selector}: expected {fld.value!r}, got {actual!r}")
            except Exception as e:
                mismatches.append(f"{fld.label or fld.selector}: unreadable ({e})")
        if mismatches:
            return WebResult(False, "Form verification failed: " + "; ".join(mismatches))
        return WebResult(True, "All filled fields verified against the DOM")

    # ------------------------------------------------------------------
    # Scraping (FR5.7)
    # ------------------------------------------------------------------

    def scrape(self, schema: dict[str, str]) -> WebResult:
        """Extract structured data. `schema` maps field name -> CSS selector.

        Selectors matching multiple elements produce multiple rows, so a table
        or result list comes back as records rather than one flat blob.
        """
        err = self._require_page()
        if err:
            return err

        for selector in schema.values():
            ok, reason = validate_selector(selector)
            if not ok:
                return WebResult(False, f"Selector rejected ({reason}): {selector[:80]}")

        columns: dict[str, list[str]] = {}
        try:
            for name, selector in schema.items():
                elements = self.page.query_selector_all(selector)
                columns[name] = [(el.inner_text() or "").strip() for el in elements]
                self.script_lines.append(
                    f"{name} = [e.inner_text() for e in page.query_selector_all({selector!r})]"
                )
        except Exception as e:
            return WebResult(False, f"Scrape failed: {e}")

        if not columns:
            return WebResult(False, "Schema produced no columns")

        row_count = max((len(v) for v in columns.values()), default=0)
        if row_count == 0:
            return WebResult(False, "No elements matched the scrape schema")

        records = [
            {name: (vals[i] if i < len(vals) else "") for name, vals in columns.items()}
            for i in range(row_count)
        ]
        return WebResult(True, f"Scraped {row_count} record(s)", data=records)

    @staticmethod
    def to_csv(records: list[dict]) -> str:
        if not records:
            return ""
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
        return buf.getvalue()

    @staticmethod
    def to_json(records: list[dict]) -> str:
        return json.dumps(records, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------

    def detect_human_gate(self) -> Optional[str]:
        """Event-response table: pause on CAPTCHA / auth wall for the human.

        We never attempt to solve these — the SDS lists CAPTCHAs as an explicit
        out-of-scope limitation.
        """
        if not self.is_running or self.page is None:
            return None
        try:
            content = (self.page.content() or "").lower()
        except Exception:
            return None
        for needle, label in (
            ("recaptcha", "reCAPTCHA challenge"),
            ("h-captcha", "hCaptcha challenge"),
            ("g-recaptcha", "reCAPTCHA challenge"),
            ("two-factor", "two-factor authentication"),
            ("verify you are human", "human verification wall"),
        ):
            if needle in content:
                return label
        return None

    def generated_script(self) -> str:
        """FR5.5: the Playwright script equivalent to everything run so far."""
        return "\n".join(self.script_lines)

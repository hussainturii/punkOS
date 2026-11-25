"""
M6: File System Intelligence Module
Local semantic file search using sentence-transformers + SQLite vector store.

Why not ChromaDB? Two reasons:
1. Lighter dependency footprint — sentence-transformers + numpy only
2. SQLite is already in our stack for history; one fewer moving part

We compute embeddings on a CPU-friendly small model (all-MiniLM-L6-v2, 22M params),
store them as BLOB columns in SQLite, and do brute-force cosine similarity at query
time. For up to ~5,000 documents this is < 100ms per query — fine for a desktop app.

Beyond ~5K docs we'd want a proper vector index (FAISS / hnswlib), but the SRS
target is 1,000 documents so brute-force is fine.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import sqlite3
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional


# ----------------------------------------------------------------------------
# Text extractors
# ----------------------------------------------------------------------------

def extract_text(path: Path, *, strict: bool = False) -> str:
    """Best-effort text extraction for common file types. Returns empty string on failure."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".txt" or suffix == ".md" or suffix == ".log":
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            return _extract_pdf(path, strict=strict)
        if suffix == ".docx":
            return _extract_docx(path, strict=strict)
        if suffix == ".csv":
            # Read first 200 lines to avoid huge files
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                return "".join(line for _, line in zip(range(200), f))
    except Exception:
        if strict:
            raise
        return ""
    return ""


def _extract_pdf(path: Path, *, strict: bool = False) -> str:
    try:
        import PyPDF2
    except ImportError:
        if strict:
            raise RuntimeError('PDF reader unavailable: install PyPDF2')
        return ""
    try:
        reader = PyPDF2.PdfReader(str(path))
        parts = []
        for page in reader.pages[:30]:    # cap at 30 pages
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    except Exception:
        if strict:
            raise
        return ""


def _extract_docx(path: Path, *, strict: bool = False) -> str:
    try:
        import docx
    except ImportError:
        if strict:
            raise RuntimeError('Word reader unavailable: install python-docx')
        return ""
    try:
        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text)
    except Exception:
        if strict:
            raise
        return ""


# ----------------------------------------------------------------------------
# Field extractors (lightweight regex — no LLM)
# ----------------------------------------------------------------------------

DATE_PATTERNS = [
    r"\b(\d{4}-\d{2}-\d{2})\b",                                     # 2024-04-15
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",                         # 4/15/24
    r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4})\b",
]
AMOUNT_PATTERN = r"\$\s*((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)(?!\d|[,.]\d)"
VENDOR_HINTS = ("from", "vendor:", "billed by", "invoice from", "supplier:")


def extract_fields(text: str) -> dict[str, Optional[str]]:
    """Extract Date, Amount, Vendor from raw text using regex. Best-effort."""
    fields: dict[str, Optional[str]] = {"date": None, "amount": None, "vendor": None}

    # Date — first match wins
    for pat in DATE_PATTERNS:
        m = re.search(pat, text)
        if m:
            fields["date"] = m.group(1)
            break

    # Amount — pick the largest dollar value (usually the total)
    matches = re.findall(AMOUNT_PATTERN, text)
    if matches:
        try:
            normalized = [float(m.replace(",", "")) for m in matches if m.replace(",", "").replace(".", "").isdigit()]
            if normalized:
                fields["amount"] = f"${max(normalized):.2f}"
        except ValueError:
            pass

    # Vendor — look for line containing a hint, take next 1-3 words
    lower = text.lower()
    for hint in VENDOR_HINTS:
        idx = lower.find(hint)
        if idx >= 0:
            after = text[idx + len(hint): idx + len(hint) + 80]
            after = after.lstrip(" :").strip()
            # Take up to first newline or period
            stop = min((after.find(c) for c in ("\n", ".", ",") if after.find(c) > 0), default=len(after))
            fields["vendor"] = after[:stop].strip() or None
            if fields["vendor"]:
                break

    return fields


def classify_file(text: str) -> str:
    """Crude rule-based classification. Returns one of:
    Invoice / Receipt / Contract / Report / Note / Other."""
    if not text:
        return "Other"
    t = text.lower()[:5000]
    if "invoice" in t or "amount due" in t:
        return "Invoice"
    if "receipt" in t and ("total" in t or "$" in t):
        return "Receipt"
    if len(text) < 50:
        return "Other"
    if "agreement" in t and ("party" in t or "parties" in t):
        return "Contract"
    if "abstract" in t or "introduction" in t and "conclusion" in t:
        return "Report"
    if len(text) < 1500:
        return "Note"
    return "Other"


# ----------------------------------------------------------------------------
# Embedding store
# ----------------------------------------------------------------------------

@dataclass
class FileResult:
    path: str
    score: float                 # cosine similarity 0-1
    snippet: str
    file_type: str
    fields: dict[str, Optional[str]]
    modified_at: str

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "score": self.score,
            "snippet": self.snippet,
            "file_type": self.file_type,
            "fields": self.fields,
            "modified_at": self.modified_at,
        }


class IndexingError(RuntimeError):
    """A partial index build; successful documents remain committed for retry."""
    def __init__(self, indexed: int, errors: list[str]):
        self.indexed = indexed
        self.errors = errors
        super().__init__(f'{len(errors)} indexing error(s): ' + '; '.join(errors[:3]))


def declared_status(text: str) -> Optional[str]:
    """Read an exact status field, not prose, filenames, or inferred authority.

    Conflicting fields remain unknown rather than silently choosing one.
    """
    values = re.findall(r'^\s*Status\s*:\s*([^\r\n]+)', text, flags=re.I | re.M)
    statuses = {value.strip().casefold() for value in values}
    supported = {'approved', 'confirmed', 'proposed', 'draft', 'rejected', 'superseded'}
    return next(iter(statuses)) if len(statuses) == 1 and statuses <= supported else None


class FileSearchEngine:
    """Indexer + searcher. Loads sentence-transformers lazily so startup is fast."""

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS files (
            path           TEXT PRIMARY KEY,
            sha256         TEXT NOT NULL,
            file_type      TEXT NOT NULL,
            modified_at    TEXT NOT NULL,
            extracted_text TEXT NOT NULL,
            snippet        TEXT NOT NULL,
            fields_json    TEXT NOT NULL,
            embedding      BLOB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_files_modified ON files(modified_at DESC);
    """

    EMBEDDING_DIM = 384         # all-MiniLM-L6-v2 output size

    def __init__(self, db_path: str | Path, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(self.SCHEMA)
        self.model_name = model_name
        self._model = None

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """Commit on clean exit and ALWAYS close — see TaskHistoryStore._conn."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @property
    def model(self):
        """Lazy-load the embedding model (slow first time, ~80MB download)."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                ) from e
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ---------------------- helpers ----------------------

    @staticmethod
    def _serialize_vec(vec) -> bytes:
        # vec is numpy float32 array of length EMBEDDING_DIM
        return struct.pack(f"{FileSearchEngine.EMBEDDING_DIM}f", *vec.tolist())

    @staticmethod
    def _deserialize_vec(data: bytes):
        import numpy as np
        return np.array(struct.unpack(f"{FileSearchEngine.EMBEDDING_DIM}f", data), dtype="float32")

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _make_snippet(text: str, max_len: int = 280) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_len] + ("…" if len(text) > max_len else "")

    # ---------------------- indexing ----------------------

    def index_directory(
        self,
        directory: str | Path,
        *,
        on_progress: Optional[Callable[[str], None]] = None,
        extensions: tuple[str, ...] = (".txt", ".md", ".pdf", ".docx", ".csv", ".log"),
    ) -> int:
        """Walk a directory and index all matching files. Skips files whose hash
        is already in the DB (cheap skip-if-unchanged).
        Returns the number of files newly indexed. Raises IndexingError with
        the successful count on partial failure, so callers cannot report success.
        """
        directory = Path(directory).expanduser().resolve()
        if not directory.is_dir():
            raise NotADirectoryError(f'Index directory is missing or not a folder: {directory}')

        new_count = 0
        errors = []
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in extensions:
                continue
            try:
                if self._index_one(path, on_progress=on_progress):
                    new_count += 1
            except Exception as e:
                errors.append(f'{path.name}: {e}')
                if on_progress:
                    on_progress(f"  ✗ {path.name}: {e}")
        if errors:
            raise IndexingError(new_count, errors)
        return new_count

    def index_files(self, paths, *, on_progress=None):
        """Index only explicitly supplied files; no directory expansion."""
        count, errors = 0, []
        for path in dict.fromkeys(Path(p) for p in paths):
            try:
                if not path.is_file():
                    raise FileNotFoundError(str(path))
                count += int(self._index_one(path, on_progress=on_progress))
            except Exception as exc:
                errors.append(f'{path.name}: {exc}')
        if errors:
            raise IndexingError(count, errors)
        return count

    def _index_one(self, path: Path, on_progress: Optional[Callable[[str], None]] = None) -> bool:
        """Index a single file. Returns True if newly indexed (or re-indexed),
        False if skipped because unchanged."""
        path_str = str(path.resolve())
        sha = self._hash_file(path)

        # Already indexed and unchanged?
        with self._conn() as c:
            row = c.execute("SELECT sha256, extracted_text, fields_json, embedding FROM files WHERE path = ?", (path_str,)).fetchone()
            if row and row["sha256"] == sha and len(row['embedding']) == self.EMBEDDING_DIM * 4:
                # Refresh cheap derived metadata after extractor repairs without
                # encoding unchanged content again.
                fields_json = _json_dumps(extract_fields(row['extracted_text']))
                if fields_json != row['fields_json']:
                    c.execute('UPDATE files SET fields_json = ? WHERE path = ?', (fields_json, path_str))
                return False

        # Extract text
        text = extract_text(path, strict=True)
        # Embed (truncate to ~2000 chars to keep things fast)
        # Empty/short documents still need filename-search entries.
        embed_text = text[:2000] or path.name
        vec = self.model.encode(embed_text, normalize_embeddings=True)

        # Field extraction
        fields = extract_fields(text)
        file_type = classify_file(text)
        snippet = self._make_snippet(text)
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()

        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO files
                   (path, sha256, file_type, modified_at, extracted_text, snippet, fields_json, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    path_str, sha, file_type, modified_at,
                    text[:50000],   # cap stored text
                    snippet,
                    _json_dumps(fields),
                    self._serialize_vec(vec),
                ),
            )
            c.commit()

        if on_progress:
            on_progress(f"  ✓ {path.name}  [{file_type}]")
        return True

    def remove_missing(self) -> int:
        """Drop entries whose underlying files no longer exist. Returns count removed."""
        removed = 0
        with self._conn() as c:
            rows = c.execute("SELECT path FROM files").fetchall()
            for r in rows:
                if not Path(r["path"]).exists():
                    c.execute("DELETE FROM files WHERE path = ?", (r["path"],))
                    removed += 1
            c.commit()
        return removed

    def file_count(self) -> int:
        with self._conn() as c:
            (n,) = c.execute("SELECT COUNT(*) FROM files").fetchone()
        return n

    # ---------------------- search ----------------------

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        min_score: float = 0.30,
        path_filter: Optional[list[str]] = None,
        status_filter: Optional[str] = None,
        search_mode: str = 'semantic',
    ) -> list[FileResult]:
        """Search indexed documents by meaning, filename or literal content.

        A declared status is source metadata, not independent verification.
        The similarity threshold applies only to semantic candidates. Changed
        files are omitted until reindexed so stale snippets are not reported.
        """
        import numpy as np

        if search_mode not in {'semantic', 'filename', 'content'}:
            raise ValueError('Unsupported search mode')
        if top_k <= 0:
            return []

        if status_filter not in (None, 'approved', 'confirmed', 'proposed', 'draft', 'rejected', 'superseded'):
            raise ValueError('Unsupported document status filter')
        if not query or not query.strip():
            return []

        with self._conn() as c:
            rows = c.execute(
                "SELECT path, sha256, file_type, modified_at, snippet, fields_json, embedding, extracted_text FROM files"
            ).fetchall()

        if not rows:
            return []

        # Optional path-prefix filter (e.g. only search Documents)
        if path_filter is not None:
            allowed = [Path(p).expanduser().resolve() for p in path_filter if str(p).strip()]
            rows = [r for r in rows if any(
                Path(r['path']).expanduser().resolve().is_relative_to(root)
                for root in allowed)]

        rows = [r for r in rows if Path(r['path']).is_file()]
        if not rows:
            return []

        # Cosine similarity (vectors already normalized)
        scored = []
        q = (np.asarray(self.model.encode(query, normalize_embeddings=True), dtype='float32')
             if search_mode == 'semantic' else None)
        needle = query.strip().casefold()
        for r in rows:
            if status_filter is not None and declared_status(r['extracted_text']) != status_filter:
                continue
            if search_mode != 'semantic':
                candidate = Path(r['path']).name if search_mode == 'filename' else r['extracted_text']
                if needle not in candidate.casefold():
                    continue
                score = 1.0 if candidate.casefold() == needle else .9
                scored.append((score, r))
                continue
            if len(r['embedding']) != self.EMBEDDING_DIM * 4:
                raise RuntimeError(
                    f"Index entry for {Path(r['path']).name} is damaged. "
                    "Click 'Build / update index' to repair it, then search again.")
            v = self._deserialize_vec(r["embedding"])
            score = float(np.dot(q, v))    # already unit vectors
            if score < min_score:
                continue
            scored.append((score, r))

        scored.sort(key=lambda x: (-x[0], x[1]['path'].casefold()))
        # Keep cosine scores intact for display. A bounded lexical bonus helps
        # distinguish topical evidence from broadly similar prose, including
        # evidence beyond the short embedding prefix. No required keywords:
        # synonym-only semantic matches remain eligible.
        terms = set(re.findall(r'\w+', query.casefold()))
        lexical = {}
        if search_mode == 'semantic' and terms:
            tokens = {r['path']: set(re.findall(r'\w+',
                      (Path(r['path']).stem + ' ' + r['extracted_text']).casefold()))
                      for _, r in scored}
            weights = {t: math.log(1 + len(tokens) / (1 + sum(t in ts for ts in tokens.values())))
                       for t in terms}
            total = sum(weights.values())
            lexical = {p: sum(weights[t] for t in terms & ts) / total
                       for p, ts in tokens.items()} if total else {}
            scored.sort(key=lambda x: (-(x[0] + .12 * lexical.get(x[1]['path'], 0)),
                                        x[1]['path'].casefold()))

        # Validate before diversifying: a stale copy must not displace a valid
        # document. Identical bytes are copies, not proof of source authority.
        current = []
        copies = []
        seen_hashes = set()
        for score, r in scored:
            try:
                if self._hash_file(Path(r['path'])) != r['sha256']:
                    continue
            except OSError:
                continue
            if search_mode == 'semantic' and r['sha256'] in seen_hashes:
                copies.append((score, r))
            else:
                current.append((score, r))
                seen_hashes.add(r['sha256'])
            if len(current) >= top_k:
                break
        scored = current + copies
        results: list[FileResult] = []
        for score, r in scored:
            if len(results) >= top_k:
                break
            snippet = r['snippet']
            if search_mode == 'semantic' and lexical.get(r['path'], 0):
                matches = list(re.finditer(r'\w+', r['extracted_text']))
                offsets = [m.start() for m in matches if m.group().casefold() in terms]
                if offsets:
                    # Prefer a window containing several query terms.
                    start = max(offsets, key=lambda pos: len(set(re.findall(
                        r'\w+', r['extracted_text'][max(0, pos-60):pos+220].casefold())) & terms))
                    snippet = self._make_snippet(r['extracted_text'][max(0, start-60):])
            results.append(FileResult(
                path=r["path"],
                score=score,
                snippet=(self._make_snippet(r['extracted_text'][max(0, r['extracted_text'].casefold().find(needle) - 60):])
                         if search_mode == 'content' else snippet),
                file_type=r["file_type"],
                fields={**_json_loads(r["fields_json"]), 'declared_status': declared_status(r['extracted_text']), 'search_mode': search_mode},
                modified_at=r["modified_at"],
            ))
        return results


# ----------------------------------------------------------------------------
# Tiny JSON helpers (avoid pulling in pydantic just for this)
# ----------------------------------------------------------------------------

def _json_dumps(obj) -> str:
    import json
    return json.dumps(obj)


def _json_loads(s: str):
    import json
    return json.loads(s) if s else {}

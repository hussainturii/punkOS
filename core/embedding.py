"""
Shared text embedder.

Used by M6 file search, M10 workflow clustering, and M11 memory retrieval so
they all share one vector space and one model download.

Primary backend is sentence-transformers/all-MiniLM-L6-v2 (22M params, the
model named in the SDS). If it is not installed, we fall back to a deterministic
hashed bag-of-words vector rather than crashing.

The fallback is genuinely weaker: it captures lexical overlap only, so
"open the calculator" and "launch calc" score poorly against each other where
the real model would match them. It exists so that memory, learning and the
test suite keep working on a machine without the ML stack — not as an
equivalent. `is_semantic` tells callers which backend is live so they can warn
the user, and `backend_name()` surfaces it in the UI.
"""
from __future__ import annotations

import hashlib
import math
import re
import struct
from typing import Optional, Sequence

EMBEDDING_DIM = 384          # all-MiniLM-L6-v2 output size
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class TextEmbedder:
    """Lazy-loading embedder with a dependency-free fallback."""

    def __init__(self, model_name: str = DEFAULT_MODEL, allow_fallback: bool = True):
        self.model_name = model_name
        self.allow_fallback = allow_fallback
        self._model = None
        self._load_attempted = False
        self._load_error: Optional[str] = None

    # ------------------------------------------------------------------

    def _load(self):
        if self._load_attempted:
            return self._model
        self._load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        except Exception as e:
            self._load_error = str(e)
            if not self.allow_fallback:
                raise RuntimeError(
                    f"Embedding model unavailable and fallback disabled: {e}"
                ) from e
            self._model = None
        return self._model

    @property
    def is_semantic(self) -> bool:
        """True when the real transformer is in use."""
        self._load()
        return self._model is not None

    def backend_name(self) -> str:
        return self.model_name if self.is_semantic else "hashed-bag-of-words (fallback)"

    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[float]:
        """Return a unit-length vector of length EMBEDDING_DIM."""
        model = self._load()
        if model is not None:
            vec = model.encode(text or "", normalize_embeddings=True)
            return [float(x) for x in vec]
        return self._hashed_embedding(text or "")

    @staticmethod
    def _hashed_embedding(text: str) -> list[float]:
        """Deterministic hashed bag-of-words with unigrams and bigrams.

        Bigrams give a little word-order sensitivity, so "move file to folder"
        and "move folder to file" are not identical vectors.
        """
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return [0.0] * EMBEDDING_DIM

        grams = list(tokens)
        grams += [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]

        vec = [0.0] * EMBEDDING_DIM
        for gram in grams:
            digest = hashlib.md5(gram.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % EMBEDDING_DIM
            # Sign from another byte so unrelated terms can cancel instead of
            # every vector drifting positive.
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign

        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            return vec
        return [x / norm for x in vec]


# ---------------------------------------------------------------------------
# Vector helpers — no numpy dependency
# ---------------------------------------------------------------------------

def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Both inputs are expected to be unit vectors, so this is a dot product."""
    if len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


def serialize_vector(vec: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def deserialize_vector(data: bytes) -> list[float]:
    count = len(data) // 4
    return list(struct.unpack(f"{count}f", data))

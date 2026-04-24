"""Shared cache naming helpers for unified evaluation pipeline."""

from __future__ import annotations

import os
from typing import Optional


CANONICAL_EVALUATION_CACHE = "evaluation_cache.json"


def canonical_cache_path(result_dir: str) -> str:
    """Return the canonical evaluation cache path."""
    return os.path.join(result_dir, CANONICAL_EVALUATION_CACHE)


def find_existing_cache(result_dir: str) -> Optional[str]:
    """Return canonical cache path if exists, else None."""
    path = canonical_cache_path(result_dir)
    return path if os.path.exists(path) else None


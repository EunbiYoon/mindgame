"""Keep HuggingFace caches on /work (cluster home quota is often tiny)."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _default_cache_root() -> Path:
    if os.environ.get("MAF_HF_CACHE"):
        return Path(os.environ["MAF_HF_CACHE"])
    return PROJECT_ROOT / "games" / ".cache" / "huggingface"


def _is_home_cache(path: Path) -> bool:
    try:
        path.resolve().relative_to((Path.home() / ".cache").resolve())
        return True
    except ValueError:
        return False


def ensure_hf_cache() -> str:
    """Point HF_HOME / datasets / hub caches at project storage."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home and not _is_home_cache(Path(hf_home)):
        cache_root = Path(hf_home)
    else:
        cache_root = _default_cache_root()

    datasets_cache = cache_root / "datasets"
    hub_cache = cache_root / "hub"
    datasets_cache.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    return str(datasets_cache)

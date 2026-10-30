"""Data locations for PAIN.

All raw inputs are read from the repository's shared data root (the same tree the other
models read), and PAIN writes its compiled path artifacts to its own subdirectory.

Environment variables:

``INV_GNN_DATA``
    Data root.  Default: ``models/PAIN/data`` (a symlink to the repository's ``data/``).
    PAIN only reads from it, except for the subdirectory below.

``PAIN_PREPROCESSED_ROOT``
    Where PAIN writes its compiled path artifacts.  Default
    ``$INV_GNN_DATA/preprocessed/pain_nc``.  No other model reads or writes it.

The Freebase variant family lives in ``dataset_variant_3hops_filtered``; the upstream
spelling ``dataset_variant_3hops_filter`` is accepted as well.
"""
from __future__ import annotations

import os
from pathlib import Path


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

# Upstream wrote everything under this CWD-relative prefix.  Config files and
# CLI defaults that still use it are re-rooted at PAIN_PREPROCESSED_ROOT.
LEGACY_PREPROCESSED_PREFIX = Path("data/preprocessed")

FREEBASE_VARIANT_DIRNAMES = (
    "dataset_variant_3hops_filtered",
    "dataset_variant_3hops_filter",
)


def data_root() -> Path:
    """Root of the shared, read-only dataset tree."""
    return Path(os.environ.get("INV_GNN_DATA", DEFAULT_DATA_ROOT)).expanduser()


def preprocessed_root() -> Path:
    """Root for PAIN's own compiled artifacts (the only tree PAIN writes)."""
    override = os.environ.get("PAIN_PREPROCESSED_ROOT")
    if override:
        return Path(override).expanduser()
    return data_root() / "preprocessed" / "pain_nc"


def raw_dir(name: str) -> Path:
    return data_root() / "raw" / name


def imdb_csv() -> Path:
    return raw_dir("IMDB") / "movie_metadata.csv"


def dblp_raw_dir() -> Path:
    return raw_dir("DBLP")


def freebase_variants_root() -> Path:
    """Directory holding ``unchanged/``, ``exact_2/``, ``exact_3/``, ``union_exact_2_3/``."""
    root = data_root()
    for name in FREEBASE_VARIANT_DIRNAMES:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return root / FREEBASE_VARIANT_DIRNAMES[0]


def preprocessed(*parts: str) -> Path:
    """A path below :func:`preprocessed_root`, e.g. ``preprocessed("IMDB")``."""
    return preprocessed_root().joinpath(*parts)


def resolve_artifact_path(value: str | Path) -> Path:
    """Map a configured artifact path onto the data root.

    Absolute paths are returned unchanged.  A path below the upstream
    ``data/preprocessed/`` prefix is re-rooted at :func:`preprocessed_root`, so
    the checked-in YAML configs keep working without edits.  Anything else stays
    relative to the current working directory.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    try:
        tail = path.relative_to(LEGACY_PREPROCESSED_PREFIX)
    except ValueError:
        return path
    return preprocessed_root() / tail


def describe() -> str:
    return (
        f"INV_GNN_DATA           {data_root()}\n"
        f"PAIN_PREPROCESSED_ROOT {preprocessed_root()}\n"
        f"IMDb csv               {imdb_csv()}\n"
        f"DBLP raw               {dblp_raw_dir()}\n"
        f"Freebase variants      {freebase_variants_root()}"
    )


if __name__ == "__main__":  # pragma: no cover - operator convenience
    print(describe())

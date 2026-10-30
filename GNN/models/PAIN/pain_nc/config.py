"""Small YAML configuration helpers."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from pain_nc.paths import resolve_artifact_path


DATA_PATH_KEYS = ("preprocessed_dir", "shared_path")


def resolve_data_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Re-root the config's artifact paths onto the data root.

    The checked-in YAML files name artifacts as ``data/preprocessed/<name>``.
    Here that prefix maps to ``PAIN_PREPROCESSED_ROOT`` (see ``pain_nc.paths``),
    so the configs keep their upstream form while the artifacts live beside
    the other models' preprocessed data.
    """
    data = config.get("data")
    if isinstance(data, dict):
        for key in DATA_PATH_KEYS:
            if data.get(key) is not None:
                data[key] = str(resolve_artifact_path(data[key]))
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return resolve_data_paths(config)


def merged_config(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    for dotted_key, value in overrides.items():
        target = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result


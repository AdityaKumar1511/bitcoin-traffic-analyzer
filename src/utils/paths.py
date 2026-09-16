"""
Project path resolution and configuration loading utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_project_root() -> Path:
    """Return the repository root directory as a Path object."""
    return PROJECT_ROOT


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Optional explicit path to config YAML file.
            Defaults to PROJECT_ROOT / "config" / "config.yaml".

    Returns:
        Dictionary with parsed configuration data.
    """
    if config_path is None:
        target_path = PROJECT_ROOT / "config" / "config.yaml"
    else:
        target_path = Path(config_path)
        if not target_path.is_absolute():
            target_path = PROJECT_ROOT / target_path

    if not target_path.is_file():
        return {}

    with target_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}

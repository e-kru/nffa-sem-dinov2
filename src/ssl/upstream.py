"""Resolve the dependency before importing any Meta components."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PIN = json.loads((ROOT / "dinov2-upstream.json").read_text())


def git_output(directory: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(directory), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def checkout_path(directory: str | Path | None = None) -> Path:
    return Path(directory or os.environ.get(
        "DINOV2_ROOT", ROOT / PIN["default_directory"]
    )).expanduser().resolve()


def activate(directory: str | Path | None = None) -> Path:
    """Fail closed on a wrong SHA, modified code, or an already shadowed import."""
    path = checkout_path(directory)
    if git_output(path, "rev-parse", "HEAD") != PIN["commit"]:
        raise RuntimeError(f"DINOv2 must be pinned to {PIN['commit']}: {path}")
    if git_output(path, "status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError(f"DINOv2 checkout must be clean: {path}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    module = importlib.import_module("dinov2")
    if Path(module.__file__).resolve().parent != path / "dinov2":
        raise RuntimeError(f"Another dinov2 package is already imported: {module.__file__}")
    return path

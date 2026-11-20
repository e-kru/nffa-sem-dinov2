"""Read-only package inventory before/after preparing an existing environment."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    prefix = Path(sys.prefix)
    packages = []
    for dist in importlib.metadata.distributions():
        record = dist.read_text("RECORD")
        packages.append({"name": dist.metadata["Name"], "version": dist.version,
                         "requires": dist.requires or [],
                         "metadata_record_sha256": hashlib.sha256(record.encode()).hexdigest() if record else None})
    packages.sort(key=lambda d: d["name"].lower())
    (args.output / "distributions.json").write_text(json.dumps(packages, indent=2) + "\n")
    (args.output / "runtime.json").write_text(json.dumps({
        "python": sys.version, "executable": sys.executable, "prefix": str(prefix),
        "host": platform.node(), "platform": platform.platform(),
    }, indent=2) + "\n")
    history = prefix / "conda-meta/history"
    if history.exists():
        shutil.copyfile(history, args.output / "conda-history.txt")
    metadata = args.output / "conda-meta"
    metadata.mkdir()
    for path in sorted((prefix / "conda-meta").glob("*.json")):
        shutil.copyfile(path, metadata / path.name)
    conda = prefix.parents[1] / "bin/conda"
    commands = {
        "pip-freeze.txt": [sys.executable, "-m", "pip", "freeze", "--all"],
        "pip-list.json": [sys.executable, "-m", "pip", "list", "--format=json"],
        "pip-check.txt": [sys.executable, "-m", "pip", "check"],
    }
    if conda.is_file():
        commands.update({
            "conda-list.json": [str(conda), "list", "--prefix", str(prefix), "--json"],
            "conda-explicit.txt": [str(conda), "list", "--prefix", str(prefix), "--explicit"],
            "conda-environment.yml": [str(conda), "env", "export", "--prefix", str(prefix)],
        })
    results = {}
    for filename, command in commands.items():
        with (args.output / filename).open("w") as stdout, (args.output / (filename + ".stderr")).open("w") as stderr:
            try:
                result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=120,
                                        env={**os.environ, "CONDA_NO_PLUGINS": "true"})
                results[filename] = {"returncode": result.returncode}
            except subprocess.TimeoutExpired:
                results[filename] = {"error": "timeout after 120 seconds"}
    (args.output / "capture-results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "packages": len(packages), "commands": results}, indent=2))


if __name__ == "__main__":
    main()

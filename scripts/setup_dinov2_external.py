"""Clone the exact upstream revision outside the project; never reset local edits."""

import argparse
from pathlib import Path
import subprocess

from src.ssl.upstream import PIN, activate, checkout_path, git_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    path = checkout_path(args.directory)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", PIN["repository"], str(path)], check=True)
    if git_output(path, "status", "--porcelain"):
        raise RuntimeError(f"Refusing to change a dirty checkout: {path}")
    subprocess.run(["git", "-C", str(path), "checkout", "--detach", PIN["commit"]], check=True)
    activate(path)
    print(f"Verified {path} at {PIN['commit']}")


if __name__ == "__main__":
    main()

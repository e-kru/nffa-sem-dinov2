"""Convenience entry point for resolution ablation A."""
import sys
from scripts.run_aachen_ablation import main

if __name__ == "__main__":
    main(["--experiment", "A", *sys.argv[1:]])

"""Convenience entry point for quadrant ablation B."""
import sys
from scripts.run_aachen_ablation import main

if __name__ == "__main__":
    main(["--experiment", "B", *sys.argv[1:]])

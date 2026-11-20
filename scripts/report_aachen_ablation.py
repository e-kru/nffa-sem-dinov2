"""Aggregate completed validation runs, plot comparisons and original-mask panels."""
import argparse
from pathlib import Path
from src.aachen_ablation.report import create_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=Path("outputs/aachen_heerlen/resolution_cropping_v1"))
    p.add_argument("--example-count", type=int, default=3)
    args = p.parse_args()
    if args.example_count < 0:
        p.error("--example-count must be nonnegative")
    create_report(args.output_root.resolve(), args.example_count)


if __name__ == "__main__":
    main()

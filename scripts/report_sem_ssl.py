"""Generate validation comparison and separate SSL/downstream plots; no evaluation."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ssl.reporting import collect, render_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--evaluation-root', type=Path, default=Path('outputs/encoder_evaluation/v1'))
    p.add_argument('--steps', type=int, nargs='+', default=[0, 50, 100, 200])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = collect(args.run, args.evaluation_root, args.steps)
    render_report(result, args.run, args.output)
    print(f"Report: {args.output / 'comparison.md'}; complete={result['complete']}")


if __name__ == '__main__':
    main()

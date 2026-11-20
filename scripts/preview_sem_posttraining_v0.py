"""CPU-only V0 crop check; does not create a model or start SSL training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--training-resolution', type=int, choices=(224, 518), default=224)
    parser.add_argument('--samples', type=int, default=2)
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error('samples must be positive')
    from src.sem_cellstyle.recipe import configuration
    from src.sem_corpus_v0.data import prepare_training_data
    cfg, dataset, _, identity = prepare_training_data(configuration(), args.manifest,
                                                      training_resolution=args.training_resolution)
    shapes = []
    for index in range(min(args.samples, len(dataset))):
        views, _ = dataset[index]
        shapes.append({key: [list(value.shape) for value in views[key]] for key in ('global_crops', 'local_crops')})
    print(json.dumps(dict(identity=identity, samples=shapes, training_started=False), indent=2))


if __name__ == '__main__':
    main()

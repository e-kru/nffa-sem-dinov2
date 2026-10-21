"""Central configuration shared by data construction and downstream experiments."""

import os
from pathlib import Path

# Override on another machine with: export NFFA_SEM_ROOT=/path/to/nffa_sem_100
DATASET_ROOT = Path(os.environ.get(
    "NFFA_SEM_ROOT", "/mnt/data/stud-ueqio/datasets/nffa_sem_100"
))

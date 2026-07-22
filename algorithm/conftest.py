"""Pin algorithm/ as the import root for the flat, non-installed layout.

Run from algorithm/: the top-level packages (models, common, utils, dataset,
engine) import directly. This conftest guarantees algorithm/ is on sys.path[0]
so pytest resolves them regardless of the invocation directory, and fixes
pytest's rootdir to algorithm/.
"""
import sys
from pathlib import Path

_ALGORITHM_ROOT = Path(__file__).resolve().parent
if str(_ALGORITHM_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALGORITHM_ROOT))

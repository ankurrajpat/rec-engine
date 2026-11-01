"""Central configuration for rec-engine."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "ml-1m"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

EMB_DIM = 64
TOP_KS = (10, 50)
TRAIN_FRAC = 0.8  # chronological split: first 80% of timeline is train
SEED = 42

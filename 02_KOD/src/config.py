"""Project-wide constants and paths."""
from pathlib import Path

SEED: int = 42

N_OUTER_REPEATS: int = 20
N_INNER_FOLDS: int = 5
N_TRIALS: int = 50
TEST_SIZE: float = 0.20
VAL_SIZE: float = 0.10  # held-out validation set for threshold optimization

VARIANCE_THRESHOLD: float = 0.01
CORRELATION_THRESHOLD: float = 0.95

MRMR_K_CANDIDATES: list[int] = [20, 50, 100]

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
RESULTS_DIR: Path = ROOT_DIR / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
TABLES_DIR: Path = RESULTS_DIR / "tables"
MODELS_DIR: Path = RESULTS_DIR / "models"
CACHE_DIR: Path = ROOT_DIR / ".cache" / "pipeline"
REPORT_DIR: Path = ROOT_DIR / "report"

NORMAL_CSV: Path = RAW_DATA_DIR / "normal_radiomics.csv"
PAPILODEM_CSV: Path = RAW_DATA_DIR / "papilodem_radiomics.csv"

BASE_MODEL_NAMES: list[str] = ["LR", "SVM", "RF", "ET", "GB", "KNN"]
ENSEMBLE_MEMBERS: list[str] = ["RF", "ET", "GB"]
ALL_MODEL_NAMES: list[str] = BASE_MODEL_NAMES + ["Ensemble"]


def ensure_dirs() -> None:
    """Create all output directories if they don't exist."""
    for d in [
        PROCESSED_DATA_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        MODELS_DIR,
        CACHE_DIR,
        REPORT_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)

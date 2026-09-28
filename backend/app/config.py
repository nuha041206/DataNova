"""
DataNova backend configuration.

Central place for paths and tunable constants so nothing is hard-coded
in the route modules. Kept deliberately simple (plain Python constants)
because this is a single-user local application; no need for env-var
management or a config library.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # datanova/backend
DATA_DIR = BASE_DIR / "data"          # uploaded + generated datasets live here
DB_DIR = BASE_DIR / "data" / "db"     # SQLite database file
REPORT_DIR = BASE_DIR / "data" / "reports"  # generated PDF reports

DB_PATH = DB_DIR / "datanova.db"

# Create folders lazily via ensure_dirs()
def ensure_dirs() -> None:
    """Create the data directories if they do not exist yet."""
    for d in (DATA_DIR, DB_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Upload / dataset limits
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 50 * 1024 * 1024   # 50 MB CSV upload limit
MAX_ROWS_FULL_GENERATION = 200_000    # safety cap on synthetic row requests
MAX_PREVIEW_ROWS = 20                 # example rows returned by preview endpoints

# Column cardinality above which a column is considered high-cardinality
HIGH_CARDINALITY_THRESHOLD = 50

# Small datasets below this many rows are flagged in the health check
MIN_ROWS_RECOMMENDED = 100

# ---------------------------------------------------------------------------
# ML utility experiment defaults
# ---------------------------------------------------------------------------
ML_TEST_SIZE = 0.2                    # hold-out fraction for the real test set
ML_RANDOM_STATE = 42                  # fixed seed for reproducibility
ML_MAX_TRAIN_ROWS = 20_000            # cap training-set size to keep it fast

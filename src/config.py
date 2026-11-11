"""
Configuration module for paths, constants, and hyperparameters.
Centralizing configuration ensures reproducible pipelines without hardcoded values.
"""
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Default PaySim dataset filename
RAW_DATA_FILE = RAW_DATA_DIR / "PS_20174392719_1491204439457_log.csv"

# Time-based splitting configuration (PaySim 'step' represents hours)
# Total steps in PaySim: 744 (31 days)
TRAIN_STEP_END = 550   # Steps 1 to 550 (~day 1 to ~day 23)
VAL_STEP_END = 650     # Steps 551 to 650 (~day 23 to ~day 27)
TEST_STEP_END = 744    # Steps 651 to 744 (~day 27 to ~day 31)

# Target and identifier columns
TARGET_COL = "isFraud"
TIME_COL = "step"

# Latency SLA (in milliseconds)
MAX_LATENCY_MS = 100

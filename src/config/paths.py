from pathlib import Path
import os

# This file can be imported from ANYWHERE in the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# data
DATA_DIR  = os.path.join(PROJECT_ROOT, 'data')

# data/index
DATA_INDEX_DIR = os.path.join(DATA_DIR, 'index')

# data/processed
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# data/raw
DATA_RAW_DIR = os.path.join(DATA_DIR, 'raw')



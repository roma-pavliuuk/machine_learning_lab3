import os
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
warnings.filterwarnings("ignore")

CONFIGS = [ROOT / "config" / "config_lr.yaml", ROOT / "config" / "config_rf.yaml"]


@pytest.fixture(scope="session")
def dataset():
    from src.data import load_dataset

    return load_dataset()

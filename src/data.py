import io
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = "data/raw"


def read_csv(name: str, rev: Optional[str] = None, **kwargs) -> pd.DataFrame:
    import dvc.api

    text = dvc.api.read(f"{RAW}/{name}", repo=str(ROOT), rev=rev, encoding="utf-8")
    return pd.read_csv(io.StringIO(text), **kwargs)


def pointer_md5(name: str) -> str:
    meta = yaml.safe_load((ROOT / RAW / f"{name}.dvc").read_text(encoding="utf-8"))
    return meta["outs"][0]["md5"]


def load_dataset(rev: Optional[str] = None):
    data = read_csv("lab2_feature_matrix.csv", rev=rev)
    return data.drop(columns="bad_credit"), data["bad_credit"].to_numpy()

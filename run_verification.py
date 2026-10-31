import hashlib
import warnings
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import ROOT, load_dataset, pointer_md5
from src.evaluation import SELECT_STEPS, build_hermetic_pipeline, load_config, selected_feature_names
from src.preprocessor import DynamicPipelineBuilder

CONFIGS = ("config/config_lr.yaml", "config/config_rf.yaml")
DATA_FILES = ("german_credit_mcar.csv", "lab2_feature_matrix.csv", "lab2_test_predictions.csv")


def test_idempotent(config_path):
    a = DynamicPipelineBuilder(config_path).build()
    b = DynamicPipelineBuilder(config_path).build()
    assert repr(a) == repr(b), "Збирання не ідемпотентне"


def test_protected_excluded(config_path):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ct = DynamicPipelineBuilder(config_path).build()
    used = {col for _, _, cols in ct.transformers for col in cols}
    leaked = used & set(cfg.get("protected_attributes", []))
    assert not leaked, f"Захищені ознаки в конвеєрі: {leaked}"


def test_dvc_lineage():
    from dvc.repo import Repo

    for name in DATA_FILES:
        actual = hashlib.md5((ROOT / "data" / "raw" / name).read_bytes()).hexdigest()
        assert actual == pointer_md5(name), f"{name}: md5 {actual} ≠ {pointer_md5(name)} у .dvc"
    with Repo(str(ROOT)) as repo:
        status = repo.status()
    assert not status, f"dvc status: {status}"


def test_protected_after_resampling(config_path):
    warnings.filterwarnings("ignore")
    cfg = load_config(config_path)
    X, y = load_dataset()
    tr, _ = next(StratifiedKFold(**cfg["cv"]["outer"]).split(X, y))
    pipe = build_hermetic_pipeline(config_path).fit(X.iloc[tr], y[tr])
    Z = X.iloc[tr]
    for name in ("prep", *SELECT_STEPS):
        Z = pipe.named_steps[name].transform(Z)
    Z_res, y_res = pipe.named_steps["resample"].fit_resample(Z, y[tr])
    names = selected_feature_names(pipe)
    protected = set(cfg["protected_attributes"])
    leaked = [n for n in names if n.split("__", 1)[1] in protected
              or any(n.split("__", 1)[1].startswith(f"{p}_") for p in protected)]
    assert not leaked, f"Захищені ознаки після ресемплінгу: {leaked}"
    assert Z_res.shape[1] == Z.shape[1] == len(names)
    added = np.bincount(y_res) - np.bincount(y[tr])
    assert added[0] == 0 and added[1] > 0, f"SMOTE додав рядки не тому класу: {added}"
    return len(names), int(added[1])


if __name__ == "__main__":
    for path in CONFIGS:
        test_idempotent(path)
        test_protected_excluded(path)
        print(f"{path}: перевірки пройдено")
    test_dvc_lineage()
    print("DVC: md5 трьох файлів збігаються з .dvc-покажчиками, dvc status чистий")
    for path in CONFIGS:
        n_features, n_synthetic = test_protected_after_resampling(path)
        print(f"{path}: після SMOTE (+{n_synthetic} синтетичних bad) — {n_features} ознак, захищених немає")

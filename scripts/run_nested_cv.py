import argparse
import os
import sys
import time
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from src.data import load_dataset
from src.evaluation import load_config, run_outer_fold

CONFIGS = {"lr": ROOT / "config" / "config_lr.yaml", "rf": ROOT / "config" / "config_rf.yaml"}
VARIANTS = {"base": dict(cost_sensitive=False, resample=True),
            "cs": dict(cost_sensitive=True, resample=True),
            "nosmote": dict(cost_sensitive=False, resample=False)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=None, help="перевизначити n_trials Optuna")
    parser.add_argument("--models", nargs="+", default=list(CONFIGS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--n-jobs", type=int, default=15)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "nested"))
    args = parser.parse_args()

    X, y = load_dataset()
    group = X[load_config(CONFIGS["lr"])["fairness"]["attribute"]].to_numpy()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tasks = []
    for model in args.models:
        cfg = load_config(CONFIGS[model])
        folds = list(StratifiedKFold(**cfg["cv"]["outer"]).split(X, y))
        for variant in args.variants:
            for k, (tr, te) in enumerate(folds):
                tasks.append((model, variant, k, tr, te))
    print(f"{len(tasks)} задач (модель × варіант × зовнішній фолд), n_jobs={args.n_jobs}", flush=True)

    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(run_outer_fold)(CONFIGS[m], X, y, group, tr, te, k, n_trials=args.trials, **VARIANTS[v])
        for m, v, k, tr, te in tasks)

    for model in args.models:
        for variant in args.variants:
            chunk = [r for (m, v, *_), r in zip(tasks, results) if m == model and v == variant]
            joblib.dump(chunk, out / f"{model}_{variant}.joblib")
            aucs = [round(r["test_auc"], 4) for r in chunk]
            print(f"{model}_{variant}: зовнішні AUC {aucs}", flush=True)
    print(f"готово за {(time.time() - t0) / 60:.1f} хв", flush=True)


if __name__ == "__main__":
    main()

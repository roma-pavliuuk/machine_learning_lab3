from __future__ import annotations

import copy
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors

from src.preprocessor import DynamicPipelineBuilder, lab2_selection

ESTIMATORS = {"logistic_regression": LogisticRegression, "random_forest": RandomForestClassifier}
RESAMPLERS = {"smote": SMOTE}


def load_config(path) -> Dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _family_entry(cfg: Dict, section: str) -> Dict:
    return cfg[section][cfg["model_family"]]


def make_estimator(spec: Dict, **overrides):
    return ESTIMATORS[spec["class"]](**{**spec.get("params", {}), **overrides})


def class_weights(costs: Dict) -> Dict[int, float]:
    return {0: 1.0, 1: costs["C10"] / costs["C01"]}


SELECT_STEPS = ("basic", "filter", "wrapper", "embedded")


def selection_steps(cfg: Dict) -> List:
    sel = cfg["selection"]
    wrapper_estimator = make_estimator(_family_entry(cfg, "wrapper_estimator"))
    l1 = LogisticRegression(l1_ratio=1.0, solver="liblinear", C=sel["embedded"]["C"], max_iter=1000)
    return [
        ("basic", lab2_selection.CorrelationFilter(**sel["basic"])),
        ("filter", lab2_selection.make_filter(sel["filter"])),
        ("wrapper", lab2_selection.make_wrapper(wrapper_estimator, sel["wrapper"])),
        ("embedded", SelectFromModel(l1, threshold=1e-10)),
    ]


def build_hermetic_pipeline(config_path, *, cost_sensitive: bool = False, resample: bool = True,
                            params: Optional[Dict] = None, spy: Optional["FitSpy"] = None) -> ImbPipeline:
    cfg = load_config(config_path)
    overrides = {"class_weight": class_weights(cfg["costs"])} if cost_sensitive else {}
    steps = []
    if spy is not None:
        steps.append(("spy", spy))
    steps += [("prep", DynamicPipelineBuilder(str(config_path)).build()), *selection_steps(cfg)]
    if resample:
        rs = cfg["resampler"]
        steps.append(("resample", RESAMPLERS[rs["name"]](**rs["params"])))
    steps.append(("model", make_estimator(_family_entry(cfg, "model"), **overrides)))
    pipe = ImbPipeline(steps)
    if params:
        pipe.set_params(**{k: v for k, v in params.items() if resample or not k.startswith("resample__")})
    return pipe


FIT_LOG: Dict[str, list] = {}


class FitSpy(BaseEstimator, TransformerMixin):
    def __init__(self, key: str = "default"):
        self.key = key

    def fit(self, X, y=None):
        FIT_LOG.setdefault(self.key, []).append(frozenset(X.index))
        return self

    def transform(self, X):
        return X


def selected_feature_names(pipe) -> List[str]:
    names = np.asarray(pipe.named_steps["prep"].get_feature_names_out())
    for step in (pipe.named_steps[s] for s in SELECT_STEPS):
        if hasattr(step, "k_feature_idx_"):
            names = names[list(step.k_feature_idx_)]
        else:
            names = names[step.get_support()]
    return [str(n) for n in names]


def total_cost(y_true, y_pred, costs: Dict) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return float(tn * costs["C00"] + fp * costs["C01"] + fn * costs["C10"] + tp * costs["C11"])


def threshold_grid(costs: Dict) -> np.ndarray:
    g = costs["threshold_grid"]
    return np.linspace(g["start"], g["stop"], g["num"])


def cost_curve(y_true, p_bad, costs: Dict) -> np.ndarray:
    return np.array([total_cost(y_true, (p_bad >= t).astype(int), costs) / len(y_true)
                     for t in threshold_grid(costs)])


def optimal_threshold(y_true, p_bad, costs: Dict) -> float:
    grid = threshold_grid(costs)
    return float(grid[int(np.argmin(cost_curve(y_true, p_bad, costs)))])


def resampling_group_audit(pipe, X_train: pd.DataFrame, y_train, group: np.ndarray) -> Dict:
    Z = X_train
    for name in ("prep", *SELECT_STEPS):
        Z = pipe.named_steps[name].transform(Z)
    sampler = clone(pipe.named_steps["resample"])
    Z_res, y_res = sampler.fit_resample(Z, y_train)
    n = len(y_train)
    minority = np.flatnonzero(np.asarray(y_train) == 1)
    synth = Z_res[n:]
    nn = NearestNeighbors(n_neighbors=2).fit(Z[minority])
    _, idx = nn.kneighbors(synth)
    parent_groups = group[minority][idx]
    before = pd.Series(group[minority]).value_counts(normalize=True)
    after = pd.Series(np.r_[group[minority], parent_groups[:, 0]]).value_counts(normalize=True)
    return {"minority_before": before.to_dict(), "minority_after": after.to_dict(),
            "n_minority_before": int(len(minority)), "n_synthetic": int(len(synth)),
            "mixed_pairs_share": float(np.mean(parent_groups[:, 0] != parent_groups[:, 1]))}


def suggest(trial, space: Dict) -> Dict:
    params = {}
    for name, d in space.items():
        if d["type"] == "int":
            params[name] = trial.suggest_int(name, d["low"], d["high"], log=d.get("log", False))
        elif d["type"] == "float":
            params[name] = trial.suggest_float(name, d["low"], d["high"], log=d.get("log", False))
        else:
            params[name] = trial.suggest_categorical(name, d["choices"])
    return params


def _inner_score(pipe, X, y, cv) -> float:
    scores = []
    for tr, va in cv.split(X, y):
        model = clone(pipe).fit(X.iloc[tr], y[tr])
        scores.append(roc_auc_score(y[va], model.predict_proba(X.iloc[va])[:, 1]))
    return float(np.mean(scores))


def run_outer_fold(config_path, X: pd.DataFrame, y: np.ndarray, group: np.ndarray, train_idx, test_idx,
                   fold: int, *, cost_sensitive: bool, resample: bool = True,
                   n_trials: Optional[int] = None) -> Dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore")
    cfg = load_config(config_path)
    space = cfg["search"]["space"][cfg["model_family"]]
    if not resample:
        space = {k: v for k, v in space.items() if not k.startswith("resample__")}
    inner = StratifiedKFold(**cfg["cv"]["inner"])
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_te, y_te = X.iloc[test_idx], y[test_idx]

    def objective(trial):
        pipe = build_hermetic_pipeline(config_path, cost_sensitive=cost_sensitive, resample=resample,
                                       params=suggest(trial, space))
        return _inner_score(pipe, X_tr, y_tr, inner)

    sampler = optuna.samplers.TPESampler(seed=cfg["search"]["seed"] + fold)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials or cfg["search"]["n_trials"])

    best = build_hermetic_pipeline(config_path, cost_sensitive=cost_sensitive, resample=resample,
                                   params=study.best_params)
    oof = cross_val_predict(best, X_tr, y_tr, cv=inner, method="predict_proba")[:, 1]
    threshold = optimal_threshold(y_tr, oof, cfg["costs"])
    best.fit(X_tr, y_tr)
    p_te = best.predict_proba(X_te)[:, 1]

    audit = DynamicPipelineBuilder.audit_trail(best.named_steps["prep"])
    result = {
        "fold": fold, "test_idx": np.asarray(test_idx), "p_test": p_te, "y_test": y_te,
        "best_params": study.best_params, "inner_best_auc": study.best_value,
        "trials": study.trials_dataframe(attrs=("number", "value", "params")),
        "threshold": threshold, "selected_features": selected_feature_names(best),
        "imputer_audit": audit.assign(fold=fold),
        "test_auc": roc_auc_score(y_te, p_te),
    }
    if resample:
        result["resampling_audit"] = resampling_group_audit(best, X_tr, y_tr, group[train_idx])
    return result


def nested_cv(config_path, X, y, group, *, cost_sensitive: bool, resample: bool = True,
              n_trials: Optional[int] = None, n_jobs: int = 5) -> List[Dict]:
    from joblib import Parallel, delayed

    cfg = load_config(config_path)
    outer = StratifiedKFold(**cfg["cv"]["outer"])
    jobs = (delayed(run_outer_fold)(config_path, X, y, group, tr, te, k, cost_sensitive=cost_sensitive,
                                    resample=resample, n_trials=n_trials)
            for k, (tr, te) in enumerate(outer.split(X, y)))
    return Parallel(n_jobs=n_jobs, backend="loky")(jobs)


def fold_metrics(results: List[Dict], costs: Dict, threshold: str = "0.5") -> pd.DataFrame:
    rows = []
    for r in results:
        t = 0.5 if threshold == "0.5" else r["threshold"]
        pred = (r["p_test"] >= t).astype(int)
        rows.append({"fold": r["fold"], "threshold": t, "roc_auc": roc_auc_score(r["y_test"], r["p_test"]),
                     "f1": f1_score(r["y_test"], pred),
                     "cost_total": total_cost(r["y_test"], pred, costs),
                     "cost_per_applicant": total_cost(r["y_test"], pred, costs) / len(pred)})
    return pd.DataFrame(rows)


def fairness_by_fold(results: List[Dict], group_all: np.ndarray, privileged, threshold: str = "0.5") -> pd.DataFrame:
    rows = []
    for r in results:
        t = 0.5 if threshold == "0.5" else r["threshold"]
        granted = r["p_test"] < t
        g = group_all[r["test_idx"]]
        priv = g == privileged
        good = r["y_test"] == 0
        rate = lambda m: granted[m].mean()
        rows.append({"fold": r["fold"], "threshold": t,
                     "grant_rate_priv": rate(priv), "grant_rate_unpriv": rate(~priv),
                     "DPR": rate(~priv) / rate(priv),
                     "EOR": rate(~priv & good) / rate(priv & good)})
    return pd.DataFrame(rows)


def pooled_predictions(results: List[Dict], n: int) -> np.ndarray:
    p = np.full(n, np.nan)
    for r in results:
        p[r["test_idx"]] = r["p_test"]
    return p

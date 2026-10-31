import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.evaluation import FIT_LOG, FitSpy, build_hermetic_pipeline, resampling_group_audit, selected_feature_names

from conftest import CONFIGS


@pytest.fixture(scope="module")
def small(dataset):
    X, y = dataset
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(X), 400, replace=False))
    return X.iloc[idx], y[idx]


def test_no_fit_sees_validation_rows(small):
    X, y = small
    FIT_LOG.pop("test", None)
    pipe = build_hermetic_pipeline(CONFIGS[0], spy=FitSpy("test"))
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")
    folds = [(set(X.index[tr]), set(X.index[va])) for tr, va in cv.split(X, y)]
    log = FIT_LOG["test"]
    assert len(log) == 3
    for seen, (train, valid) in zip(log, folds):
        assert seen == train and not (seen & valid)
    assert proba.shape == (len(X), 2)


def test_selection_is_refit_per_fold(small):
    X, y = small
    cv = StratifiedKFold(2, shuffle=True, random_state=1)
    selections = []
    for tr, _ in cv.split(X, y):
        pipe = build_hermetic_pipeline(CONFIGS[0]).fit(X.iloc[tr], y[tr])
        selections.append(tuple(selected_feature_names(pipe)))
    assert all(len(s) > 0 for s in selections)
    assert selections[0] != selections[1]


def test_resampling_audit_shares(small):
    X, y = small
    pipe = build_hermetic_pipeline(CONFIGS[0]).fit(X, y)
    audit = resampling_group_audit(pipe, X, y, X["sex"].to_numpy())
    assert audit["n_synthetic"] == int((y == 0).sum() - (y == 1).sum())
    assert abs(sum(audit["minority_before"].values()) - 1) < 1e-9
    assert abs(sum(audit["minority_after"].values()) - 1) < 1e-9

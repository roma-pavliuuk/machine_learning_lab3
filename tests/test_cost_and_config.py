import ast

import numpy as np
import pytest

from src.evaluation import (class_weights, cost_curve, load_config, optimal_threshold, threshold_grid,
                            total_cost)

from conftest import CONFIGS, ROOT

COSTS = {"C00": 0, "C01": 150, "C10": 1500, "C11": 0, "threshold_grid": {"start": 0, "stop": 1, "num": 101}}


def test_total_cost_formula():
    y = np.array([0, 0, 1, 1, 1])
    pred = np.array([0, 1, 0, 1, 0])
    assert total_cost(y, pred, COSTS) == 150 + 2 * 1500


def test_optimal_threshold_matches_bayes_rule_for_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    t = optimal_threshold(y, p, COSTS)
    assert abs(t - 150 / 1650) < 0.02


def test_cost_curve_endpoints():
    y = np.array([0, 0, 0, 1])
    curve = cost_curve(y, np.array([0.1, 0.2, 0.3, 0.9]), COSTS)
    grid = threshold_grid(COSTS)
    assert curve[0] == 3 * 150 / 4
    assert curve[-1] == pytest.approx(1500 / 4) and grid[-1] == 1.0


def test_class_weights_ratio():
    assert class_weights(COSTS) == {0: 1.0, 1: 10.0}


def test_configs_differ_only_in_model_family():
    lr, rf = (p.read_text(encoding="utf-8").splitlines() for p in CONFIGS)
    diff = [(a, b) for a, b in zip(lr, rf) if a != b]
    assert len(lr) == len(rf) and len(diff) == 1 and diff[0][0].startswith("model_family:")


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_search_space_targets_existing_steps(path):
    from src.evaluation import build_hermetic_pipeline

    cfg = load_config(path)
    params = build_hermetic_pipeline(path).get_params()
    for name in cfg["search"]["space"][cfg["model_family"]]:
        assert name in params, name


def test_no_model_family_literals_in_evaluation():
    families = set(load_config(CONFIGS[0])["transform_policy"])
    tree = ast.parse((ROOT / "src" / "evaluation.py").read_text(encoding="utf-8"))
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not strings & families

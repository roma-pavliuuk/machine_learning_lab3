import hashlib
import re

import pytest

from src.data import ROOT, pointer_md5, read_csv
from src.preprocessor import LAB2_STEPS, MODULES, DynamicPipelineBuilder, file_sha256, lab1

DATA_FILES = ["german_credit_mcar.csv", "lab2_feature_matrix.csv", "lab2_test_predictions.csv"]


@pytest.mark.parametrize("name", list(MODULES))
def test_previous_lab_modules_unchanged(name):
    path, expected = MODULES[name]
    assert file_sha256(path) == expected, f"{path} змінено — код ЛР №1–2 переноситься без змін логіки"


def test_builder_is_lab1_class_with_lab2_steps():
    assert DynamicPipelineBuilder is lab1.DynamicPipelineBuilder
    assert all(name in lab1._STEP_FACTORIES for name in LAB2_STEPS)


@pytest.mark.parametrize("name", DATA_FILES)
def test_dvc_pointer_matches_file(name):
    actual = hashlib.md5((ROOT / "data" / "raw" / name).read_bytes()).hexdigest()
    assert actual == pointer_md5(name)


def test_dvc_api_read_matches_tabular_source(dataset):
    X, y = dataset
    german = read_csv("german_credit_mcar.csv")
    assert len(german) == len(X) == 1000
    assert (german["bad_credit"].to_numpy() == y).all()
    assert german["credit_amount"].equals(X["credit_amount"])


def test_no_direct_reads_of_raw_data():
    offenders = []
    for path in list((ROOT / "src").glob("*.py")) + list((ROOT / "notebooks").glob("*.ipynb")) + \
            list((ROOT / "scripts").glob("*.py")) + [ROOT / "run_verification.py"]:
        text = path.read_text(encoding="utf-8")
        if re.search(r"read_csv\([^)]*data/raw", text) or re.search(r"open\([^)]*data/raw[^)]*\.csv", text):
            offenders.append(path.name)
    assert not offenders, offenders

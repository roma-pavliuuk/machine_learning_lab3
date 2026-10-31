import hashlib
import importlib.util
import os
import sys
from pathlib import Path

from sklearn.preprocessing import PowerTransformer

_HERE = Path(__file__).resolve().parents[2]
LAB1_DIR = Path(os.environ.get("LAB1_DIR", _HERE / "lab_01"))
LAB2_DIR = Path(os.environ.get("LAB2_DIR", _HERE / "lab_02"))

MODULES = {
    "lab01_dynamic_pipeline_builder": (LAB1_DIR / "src" / "dynamic_pipeline_builder.py",
                                       "d0b8c633121e08f92e3f2435c2c5851b966967e83c78693155d5531a0b630b08"),
    "lab02_features": (LAB2_DIR / "src" / "features.py",
                       "0a978a73c2fa2b3467b9c6ad98ecb7b15247741b7c1fdac573b9387a71f01eb1"),
    "lab02_extraction": (LAB2_DIR / "src" / "extraction.py",
                         "36e0a037d891994288f37b64401b6b1347e13f4265ae549db49979d76156a498"),
    "lab02_selection": (LAB2_DIR / "src" / "selection.py",
                        "2a8013f07c70ae99a9094d89227ccba0f8a6d6f8cdc850b8c48c9892f6623cf7"),
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    path, _ = MODULES[name]
    if not path.exists():
        raise FileNotFoundError(f"Не знайдено {path}. Покладіть lab_01 і lab_02 поруч із lab_03 "
                                f"або задайте LAB1_DIR / LAB2_DIR.")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lab1 = _load("lab01_dynamic_pipeline_builder")
lab2_features = _load("lab02_features")
lab2_extraction = _load("lab02_extraction")
lab2_selection = _load("lab02_selection")

LAB2_STEPS = {
    "cyclic_time": lambda **p: lab2_features.CyclicTimeTransformer(**p),
    "interactions": lambda **p: lab2_features.InteractionTransformer(**p),
    "oof_woe": lambda **p: lab2_features.OOFWoEEncoder(**p),
    "yeo_johnson": lambda **p: PowerTransformer(method="yeo-johnson", **p),
    "tfidf_svd": lambda **p: lab2_extraction.TextExtractor(**p),
}
for _name, _factory in LAB2_STEPS.items():
    lab1._STEP_FACTORIES.setdefault(_name, _factory)

DynamicPipelineBuilder = lab1.DynamicPipelineBuilder
AdaptiveImputer = lab1.AdaptiveImputer
validate_config = lab1.validate_config

"""Test scaffold and configs."""

import pathlib

import yaml

import readmit


def test_version():
    assert readmit.__version__ == "0.1.0"


def test_configs_exist():
    for cfn in ["base.yaml", "features.yaml", "models.yaml"]:
        p = pathlib.Path("configs") / cfn
        assert p.exists(), f"Missing config {cfn}"
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            assert data is not None, f"Empty config {cfn}"

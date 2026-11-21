"""Configuration validation and dataset generator reproducibility."""

import hashlib
import json

import pandas as pd
import pytest

from simulation.config import load_config
from simulation.data_generator import OUTPUT_FILES, generate_dataset
from simulation.exceptions import ConfigError


# ------------------------------------------------------------------ config
@pytest.mark.parametrize("size", [19, 31])
def test_fleet_size_bounds(size):
    with pytest.raises(ConfigError, match="fleet.size"):
        load_config(overrides={"fleet": {"size": size}})


def test_overlapping_windows_rejected():
    windows = [{"id": "A", "start": "05:00", "end": "09:00", "max_inductions": 5},
               {"id": "B", "start": "08:00", "end": "12:00", "max_inductions": 5}]
    with pytest.raises(ConfigError, match="overlap"):
        load_config(overrides={"depot": {"induction_windows": windows}})


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_clock_values_converted(cfg):
    assert cfg["simulation"]["service_start"] == 5 * 3600
    assert all(isinstance(w["start"], int) for w in cfg["depot"]["induction_windows"])


# --------------------------------------------------------- reproducibility
def _hashes(folder):
    return {f: hashlib.sha256((folder / f).read_bytes()).hexdigest() for f in OUTPUT_FILES}


def test_same_seed_gives_identical_csv_files(cfg, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate_dataset(cfg, seed=7, n_days=3, output_dir=a)
    generate_dataset(cfg, seed=7, n_days=3, output_dir=b)
    assert _hashes(a) == _hashes(b)
    assert (a / "dataset_manifest.json").read_text() == (b / "dataset_manifest.json").read_text()


def test_different_seed_gives_different_data(cfg):
    a = generate_dataset(cfg, seed=7, n_days=3)
    b = generate_dataset(cfg, seed=8, n_days=3)
    assert not a["passenger_demand.csv"].equals(b["passenger_demand.csv"])
    assert not a["train_readiness.csv"].equals(b["train_readiness.csv"])


def test_full_30_day_dataset(cfg, tmp_path):
    data = generate_dataset(cfg, seed=42, n_days=30, output_dir=tmp_path)
    m = json.loads((tmp_path / "dataset_manifest.json").read_text())
    assert m["n_days"] == 30 and m["safety_violations"] == 0
    assert "NOT actual Kochi Metro" in m["notice"]
    assert all(m["scenario_counts"][t] >= 1 for t in ("normal", "peak_demand", "disruption"))
    # event tables (maintenance, delays) may legitimately have days with no events
    every_day = {"scenarios.csv", "passenger_demand.csv", "train_readiness.csv", "train_availability.csv",
                 "fleet_status_by_interval.csv", "induction_windows.csv", "track_availability.csv",
                 "platform_occupancy.csv", "induction_history.csv", "track_occupancy_log.csv"}
    for f in OUTPUT_FILES:
        df = pd.read_csv(tmp_path / f)
        assert len(df) > 0, f
        if f in every_day:
            assert df["day"].nunique() == 30, f
    demand = data["passenger_demand.csv"]
    assert demand.groupby("day").size().nunique() == 1

"""Shared pytest fixtures."""

import sys
from pathlib import Path

import numpy as np
import pytest
import simpy

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.config import load_config  # noqa: E402
from simulation.scenario_generator import ScenarioGenerator  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture
def env():
    return simpy.Environment(initial_time=0)


@pytest.fixture
def rng():
    return np.random.default_rng(123)


@pytest.fixture(scope="session")
def scenarios(cfg):
    """30 scenarios from seed 42 (contains every scenario type)."""
    return ScenarioGenerator(cfg, 42).generate(30)


def first_of_type(scenarios, scenario_type):
    return next(s for s in scenarios if s.scenario_type == scenario_type)

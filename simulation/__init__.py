"""MetroSched simulation package (Phase 1).

All data produced by this package is SYNTHETIC and is not actual Kochi Metro
Rail Limited (KMRL) operational data.
"""

from .config import DEFAULT_CONFIG_PATH, load_config
from .demand_generator import DemandGenerator, required_trains_for_demand
from .depot import Depot
from .metro_environment import MetroEnvironment, SimulationResult, readiness_order_policy
from .scenario_generator import Scenario, ScenarioGenerator, generate_fleet
from .track import PlatformGroup, TrackSection, audit_occupancy
from .train import Train, TrainLocation, TrainStatus

__all__ = [
    "DEFAULT_CONFIG_PATH", "load_config", "DemandGenerator", "required_trains_for_demand", "Depot",
    "MetroEnvironment", "SimulationResult", "readiness_order_policy", "Scenario", "ScenarioGenerator",
    "generate_fleet", "PlatformGroup", "TrackSection", "audit_occupancy", "Train", "TrainLocation",
    "TrainStatus",
]

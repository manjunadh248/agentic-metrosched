"""Train readiness validation and status state machine."""

import pytest

from simulation.exceptions import IllegalStateTransitionError, TrainNotReadyError
from simulation.train import Train, TrainLocation, TrainStatus


def make_train(**kw):
    base = dict(train_id="TS01", depot_location="SL01-A", readiness_time=1000, min_prep_time_s=600)
    base.update(kw)
    return Train(**base)


def test_not_ready_before_readiness_time():
    tr = make_train()
    assert not tr.is_ready(999)
    with pytest.raises(TrainNotReadyError, match="preparation incomplete"):
        tr.validate_ready(999)
    assert tr.is_ready(1000)
    tr.validate_ready(1000)


def test_maintenance_required_blocks_induction_and_sets_status():
    tr = make_train(maintenance_required=True)
    assert tr.status is TrainStatus.MAINTENANCE
    assert "maintenance pending" in tr.readiness_issues(10_000)
    with pytest.raises(TrainNotReadyError):
        tr.validate_ready(10_000)


def test_train_outside_depot_is_not_ready():
    tr = make_train()
    tr.transition(TrainStatus.INDUCTED, 1000, TrainLocation.TRANSIT)
    issues = tr.readiness_issues(2000)
    assert any("not in depot" in i for i in issues)
    assert any("cannot be inducted" in i for i in issues)


def test_delayed_train_in_depot_can_still_be_inducted_once_ready():
    tr = make_train()
    tr.transition(TrainStatus.DELAYED, 500, TrainLocation.DEPOT)
    assert not tr.is_ready(900)
    assert tr.is_ready(1000)
    tr.transition(TrainStatus.INDUCTED, 1000, TrainLocation.TRANSIT)


def test_legal_lifecycle_and_history():
    tr = make_train()
    tr.transition(TrainStatus.INDUCTED, 1000, TrainLocation.TRANSIT)
    tr.transition(TrainStatus.RUNNING, 1300, TrainLocation.MAINLINE)
    tr.transition(TrainStatus.DELAYED, 2000)
    tr.transition(TrainStatus.RUNNING, 2100)
    tr.transition(TrainStatus.AVAILABLE, 5000, TrainLocation.DEPOT)
    assert [h[2] for h in tr.history] == ["Inducted", "Running", "Delayed", "Running", "Available"]


@pytest.mark.parametrize("start,target", [
    (TrainStatus.AVAILABLE, TrainStatus.RUNNING),       # must be inducted first
    (TrainStatus.MAINTENANCE, TrainStatus.INDUCTED),    # must be released first
    (TrainStatus.INDUCTED, TrainStatus.AVAILABLE),
])
def test_illegal_transitions_rejected(start, target):
    tr = make_train()
    tr.status = start
    with pytest.raises(IllegalStateTransitionError):
        tr.transition(target, 0)


def test_invalid_construction():
    with pytest.raises(ValueError):
        make_train(train_id="")
    with pytest.raises(ValueError):
        make_train(induction_priority=0)
    with pytest.raises(ValueError):
        make_train(readiness_time=-1)

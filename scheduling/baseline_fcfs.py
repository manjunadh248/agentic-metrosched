"""Naive First-Come, First-Served (FCFS) baseline.

Algorithm (per decision time):

1. Identify available trains: those in the depot, prepared, not in maintenance.
2. Order them by readiness time (first ready = first served).
3. If fewer trains are committed to service than are required *right now*,
   induct the earliest-ready train into the earliest available slot (the
   environment picks the exit track / entry point that clears first).
4. Basic safety and resource constraints are checked by the environment
   (readiness, induction capacity, block occupancy, headway, platforms).
5. No look-ahead: future demand and future congestion are ignored. Before
   05:00 the opening requirement is used (the first timetable is known).

This is the benchmark the intelligent agent is evaluated against.
"""

from __future__ import annotations

from simulation.config import seconds_to_clock
from simulation.metro_environment import MetroEnvironment

from .base import Decision, Scheduler, ready_by_readiness


class FCFSScheduler(Scheduler):
    name = "FCFS"

    def decide(self, env: MetroEnvironment) -> Decision | None:
        t = env.now()
        required = env.required_at(max(t, env.service_start))
        committed = env.committed_trains()
        if committed >= required:
            return None
        if env.depot.remaining_capacity(t) <= 0:
            return None
        ready = ready_by_readiness(env)
        if not ready:
            return None
        train = ready[0]
        return Decision(
            train.train_id, None,
            f"FCFS: {train.train_id} was the first train ready (since {seconds_to_clock(train.readiness_time)}); "
            f"service needs {required} trains, {committed} committed.")

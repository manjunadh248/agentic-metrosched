"""Simple rule-based heuristic scheduler (second comparison point).

Rules (per decision time):

1. **Anticipate demand** – target = max required trains over the next
   ``lookahead_s`` seconds (covers the induction route time).
2. **Avoid bunching** – if ``max_queued_before_hold`` trains are already
   queued on the induction route, hold, unless service is short right now.
3. **Prefer healthy trains** – among ready trains choose the best induction
   priority (lowest km since maintenance), then earliest readiness.
4. **Pick the entry point** that will be clear soonest at the moment the
   train reaches the mainline.
"""

from __future__ import annotations

from simulation.config import seconds_to_clock
from simulation.metro_environment import MetroEnvironment

from .base import Decision, Scheduler


class HeuristicScheduler(Scheduler):
    name = "Heuristic"

    def __init__(self, lookahead_s: int = 720, max_queued_before_hold: int = 1, prefer_priority: bool = True):
        if lookahead_s < 0 or max_queued_before_hold < 0:
            raise ValueError("lookahead_s and max_queued_before_hold must be >= 0")
        self.lookahead_s = int(lookahead_s)
        self.max_queued = int(max_queued_before_hold)
        self.prefer_priority = bool(prefer_priority)

    @classmethod
    def from_config(cls, rl_cfg: dict) -> "HeuristicScheduler":
        h = rl_cfg["heuristic"]
        return cls(h["lookahead_s"], h["max_queued_before_hold"], h["prefer_priority"])

    def target(self, env: MetroEnvironment, t: int) -> int:
        start = max(t, env.service_start)
        end = min(start + self.lookahead_s, env.service_end - 1)
        return max(env.required_at(x) for x in range(start, end + 1, 60))

    def decide(self, env: MetroEnvironment) -> Decision | None:
        t = env.now()
        target = self.target(env, t)
        committed = env.committed_trains()
        if committed >= target or env.depot.remaining_capacity(t) <= 0:
            return None
        short_now = env.in_service_count() < env.required_at(max(t, env.service_start)) and t >= env.service_start
        if len(env.waiting) >= self.max_queued and not short_now:
            return None
        ready = env.ready_candidates()
        if not ready:
            return None
        key = (lambda tr: (tr.induction_priority, tr.readiness_time, tr.train_id)) if self.prefer_priority \
            else (lambda tr: (tr.readiness_time, tr.train_id))
        train = min(ready, key=key)
        offset = env.nominal_route_s - env.platforms.dwell_s
        entry = min(env.entry_points, key=lambda e: (e.earliest_entry(t + offset), e.queue_length(), e.section_id))
        return Decision(
            train.train_id, entry.section_id,
            f"Heuristic: need {target} trains within {self.lookahead_s // 60} min, {committed} committed; "
            f"{train.train_id} has best priority {train.induction_priority} "
            f"({train.km_since_maintenance:.0f} km since maintenance, ready {seconds_to_clock(train.readiness_time)}); "
            f"{entry.section_id} clears soonest.")

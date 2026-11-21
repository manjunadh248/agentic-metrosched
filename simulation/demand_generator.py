"""Synthetic passenger demand and the demand -> required-trains mapping.

Demand model (per interval, passengers per hour per direction, "pphpd")::

    shape(t)  = ramp(t) + sum_k amplitude_k * exp(-(t - center_k)^2 / (2 * width_k^2))
    demand(t) = base * shape(t) * weekday_factor * scenario_multiplier * LogNormal(0, sigma)

``ramp(t)`` rises linearly from ``early_morning_factor`` at service start to 1
by 07:00 and falls to ``late_evening_factor`` by service end (from 21:00).

Required trains in service::

    headway_needed = 3600 * train_capacity * target_load_factor / demand
    headway        = clip(headway_needed, min_service_headway_s, max_service_headway_s)
    cycle_time     = 2 * (one_way_run_time_s + terminal_turnaround_s)
    required       = clip(ceil(cycle_time / headway), 1, fleet_size - reserve_trains)

so a busier interval needs a shorter headway and therefore more trains in
circulation. The max_service_headway_s cap guarantees a minimum service level
even when demand is very low.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from .config import seconds_to_clock

logger = logging.getLogger(__name__)


def cycle_time_s(cfg: dict) -> int:
    svc = cfg["service"]
    return int(2 * (svc["one_way_run_time_s"] + svc["terminal_turnaround_s"]))


def required_trains_for_demand(demand_pphpd: float, cfg: dict) -> tuple[int, int]:
    """Return ``(required_trains, service_headway_s)`` for a demand level."""
    svc = cfg["service"]
    if demand_pphpd <= 0:
        headway = svc["max_service_headway_s"]
    else:
        needed = 3600.0 * cfg["fleet"]["train_capacity"] * svc["target_load_factor"] / demand_pphpd
        headway = float(np.clip(needed, svc["min_service_headway_s"], svc["max_service_headway_s"]))
    max_service = cfg["fleet"]["size"] - svc["reserve_trains"]
    required = int(np.clip(math.ceil(cycle_time_s(cfg) / headway), 1, max_service))
    return required, int(round(headway))


class DemandGenerator:
    """Produce the demand profile of one operating day."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dcfg = cfg["demand"]
        self.sim = cfg["simulation"]

    def shape(self, t: np.ndarray) -> np.ndarray:
        """Deterministic, noise-free demand multiple of base at second-of-day ``t``."""
        t = np.asarray(t, dtype=float)
        start, end = self.sim["service_start"], self.sim["service_end"]
        seven, nine_pm = 7 * 3600, 21 * 3600
        early, late = self.dcfg["early_morning_factor"], self.dcfg["late_evening_factor"]
        ramp = np.ones_like(t)
        if seven > start:
            m = t < seven
            ramp[m] = early + (1 - early) * (t[m] - start) / (seven - start)
        if end > nine_pm:
            m = t > nine_pm
            ramp[m] = 1 - (1 - late) * (t[m] - nine_pm) / (end - nine_pm)
        ramp = np.clip(ramp, min(early, late), 1.0)
        bumps = np.zeros_like(t)
        for p in self.dcfg["peaks"]:
            bumps += p["amplitude"] * np.exp(-((t - p["center"]) ** 2) / (2 * p["width_s"] ** 2))
        return ramp + bumps

    def period_label(self, t: int) -> str:
        """Name of the dominant peak near ``t`` or 'off_peak'."""
        best, best_val = "off_peak", 0.0
        for p in self.dcfg["peaks"]:
            val = p["amplitude"] * math.exp(-((t - p["center"]) ** 2) / (2 * p["width_s"] ** 2))
            if val > best_val:
                best, best_val = p["name"], val
        return best if best_val >= 0.25 else "off_peak"

    def generate(self, rng: np.random.Generator, weekday: int, demand_multiplier: float = 1.0,
                 day: int = 0) -> pd.DataFrame:
        """Demand per interval for one day.

        Args:
            rng: seeded generator (the only source of randomness).
            weekday: 0 = Monday .. 6 = Sunday.
            demand_multiplier: scenario multiplier (e.g. 1.3 on peak-demand days).
            day: day index stored in the output.
        """
        if not 0 <= weekday <= 6:
            raise ValueError("weekday must be in 0..6")
        step = int(self.sim["interval_s"])
        starts = np.arange(self.sim["service_start"], self.sim["service_end"], step)
        mids = starts + step / 2
        noise = rng.lognormal(mean=0.0, sigma=self.dcfg["noise_sigma"], size=len(starts))
        factor = self.dcfg["weekday_factors"][weekday] * demand_multiplier
        demand = self.dcfg["base_pphpd"] * self.shape(mids) * factor * noise
        rows = []
        thresh = self.dcfg["peak_threshold_factor"] * self.dcfg["base_pphpd"]
        for s, d, mid in zip(starts, demand, mids):
            req, hw = required_trains_for_demand(float(d), self.cfg)
            rows.append({
                "day": day,
                "interval_start_s": int(s),
                "interval_start": seconds_to_clock(s),
                "period": self.period_label(int(mid)),
                "is_peak": bool(d >= thresh),
                "demand_pphpd": int(round(d)),
                "service_headway_s": hw,
                "required_trains": req,
            })
        df = pd.DataFrame(rows)
        logger.debug("Day %s demand: peak %s pphpd, max %s trains", day, df["demand_pphpd"].max(),
                     df["required_trains"].max())
        return df


def required_trains_at(demand_df: pd.DataFrame, t: int) -> int:
    """Look up required trains for second-of-day ``t`` (clamped to the day)."""
    starts = demand_df["interval_start_s"].to_numpy()
    idx = int(np.searchsorted(starts, t, side="right") - 1)
    idx = min(max(idx, 0), len(starts) - 1)
    return int(demand_df["required_trains"].iloc[idx])

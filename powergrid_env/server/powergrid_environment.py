"""
PowerGrid microgrid balancing environment.

Simulates a residential microgrid over a 24-hour horizon (96 × 15-min steps).
The agent manages a battery and flexible loads to minimise energy costs while
keeping the grid stable and maximising renewable utilisation.

Physics summary
---------------
  P_solar + P_battery_discharge + P_grid_import
      = P_load_served + P_battery_charge + P_grid_export

All power in kW; energy in kWh; dt = 0.25 h per step.
"""

from __future__ import annotations

import math
import random
import uuid
from typing import Tuple

try:
    from ..models import GridAction, GridObservation, GridState
except ImportError:
    from models import GridAction, GridObservation, GridState


# ---------------------------------------------------------------------------
# Microgrid parameters
# ---------------------------------------------------------------------------
BATTERY_CAPACITY_KWH: float = 50.0       # Total usable capacity
BATTERY_MAX_POWER_KW: float = 25.0       # Max charge or discharge rate
BATTERY_EFFICIENCY: float = 0.95         # Round-trip √η per half-trip
BATTERY_SOC_MIN: float = 0.10            # Hard lower bound (battery health)
BATTERY_SOC_MAX: float = 0.95            # Hard upper bound

SOLAR_PEAK_KW: float = 20.0             # Peak PV output at solar noon
GRID_MAX_IMPORT_KW: float = 50.0        # Distribution-level limit
GRID_EXPORT_PRICE_RATIO: float = 0.40   # Export price = ratio × import price

FLEXIBLE_LOAD_FRACTION: float = 0.30    # 30 % of load is deferrable
CURTAILMENT_PENALTY_PER_KWH: float = 0.50  # $/kWh shed (discomfort cost)

STEPS_PER_DAY: int = 96                 # 24 h × 4 steps/h
DT: float = 0.25                        # hours per step

# Reward normalisation — worst-case net cost per step (max import x peak price x dt)
_MAX_STEP_COST: float = GRID_MAX_IMPORT_KW * 0.30 * DT  # 50 * 0.30 * 0.25 = 3.75

# Time-of-use price tiers  ($/kWh)
_PRICE_OFF_PEAK: float = 0.08
_PRICE_MID_PEAK: float = 0.15
_PRICE_ON_PEAK: float = 0.30


# ---------------------------------------------------------------------------
# Profile helpers (deterministic + optional noise)
# ---------------------------------------------------------------------------

def _hour(step: int) -> float:
    """Convert step index to fractional hour (0–24)."""
    return step * DT  # DT = 0.25 h → step 0 = 0:00, step 95 = 23:45


def _solar_profile(step: int, noise: float = 0.0) -> float:
    """Gaussian solar irradiance curve, zero outside daylight (6–18 h)."""
    h = _hour(step)
    if h < 6.0 or h > 18.0:
        return 0.0
    base = SOLAR_PEAK_KW * math.exp(-((h - 12.0) ** 2) / 8.0)
    return max(0.0, base * (1.0 + noise * random.gauss(0, 1)))


def _load_profile(step: int, noise: float = 0.0) -> float:
    """Bimodal residential load: morning rise + evening peak."""
    h = _hour(step)
    morning = 8.0 * math.exp(-((h - 8.0) ** 2) / 2.0)
    evening = 12.0 * math.exp(-((h - 19.0) ** 2) / 2.0)
    base = 3.0
    total = base + morning + evening
    return max(0.5, total * (1.0 + noise * random.gauss(0, 0.5)))


def _price(step: int) -> float:
    """Simple time-of-use tariff."""
    h = _hour(step)
    if (7.0 <= h < 11.0) or (17.0 <= h < 21.0):
        return _PRICE_ON_PEAK
    if (11.0 <= h < 17.0):
        return _PRICE_MID_PEAK
    return _PRICE_OFF_PEAK


def _context_text(
    step: int,
    hour: float,
    solar_kw: float,
    demand_kw: float,
    flex_kw: float,
    battery_soc: float,
    price: float,
    episode_cost: float,
) -> str:
    """Human-readable summary injected into the observation for LLM agents."""
    price_label = (
        "ON-PEAK" if price >= _PRICE_ON_PEAK
        else "MID-PEAK" if price >= _PRICE_MID_PEAK
        else "OFF-PEAK"
    )
    h = int(hour)
    m = int((hour - h) * 60)
    return (
        f"Time {h:02d}:{m:02d} | Step {step+1}/96 | "
        f"Solar {solar_kw:.1f} kW | Demand {demand_kw:.1f} kW "
        f"(flex {flex_kw:.1f} kW) | "
        f"Battery SoC {battery_soc*100:.0f}% | "
        f"Price {price:.2f} $/kWh ({price_label}) | "
        f"Cumulative cost ${episode_cost:.2f}"
    )


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class PowerGridEnvironment:
    """
    OpenEnv-compatible microgrid environment.

    reset() → GridObservation
    step(action) → (GridObservation, float, bool, dict)
    state() → GridState
    """

    def __init__(
        self,
        profile_noise: float = 0.05,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            profile_noise: Fractional Gaussian noise added to solar/load profiles.
                           Set to 0 for a fully deterministic episode.
            seed:          Random seed for reproducibility.
        """
        self.profile_noise = profile_noise
        if seed is not None:
            random.seed(seed)

        # Internal state (initialised by reset)
        self._episode_id: str = ""
        self._step: int = 0
        self._battery_soc: float = 0.5
        self._episode_cost: float = 0.0
        self._episode_curtailed_kwh: float = 0.0
        self._episode_solar_used_kwh: float = 0.0
        self._done: bool = False

        # Pre-compute profiles for the current episode
        self._solar: list[float] = []
        self._load: list[float] = []
        self._price: list[float] = []
        self._soc_trajectory: list[float] = []

    # ------------------------------------------------------------------
    def reset(self) -> GridObservation:
        self._episode_id = str(uuid.uuid4())
        self._step = 0
        self._battery_soc = random.uniform(0.2, 0.8)
        self._episode_cost = 0.0
        self._episode_curtailed_kwh = 0.0
        self._episode_solar_used_kwh = 0.0
        self._done = False
        self._soc_trajectory = []

        # Generate fresh profiles with noise
        self._solar = [_solar_profile(t, self.profile_noise) for t in range(STEPS_PER_DAY)]
        self._load = [_load_profile(t, self.profile_noise) for t in range(STEPS_PER_DAY)]
        self._price = [_price(t) for t in range(STEPS_PER_DAY)]

        obs = self._make_obs(
            battery_power_kw=0.0,
            grid_import_kw=0.0,
            grid_export_kw=0.0,
            reward=0.0,
            r_cost=0.0,
            r_curtail=0.0,
            r_renewable=0.0,
            r_stability=0.0,
        )
        return obs

    # ------------------------------------------------------------------
    def step(self, action: GridAction) -> Tuple[GridObservation, float, bool, dict]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        t = self._step
        solar_kw = self._solar[t]
        load_kw = self._load[t]
        price = self._price[t]
        flex_load = load_kw * FLEXIBLE_LOAD_FRACTION

        # ---- Clamp action ----
        battery_cmd = float(max(-1.0, min(1.0, action.battery_action)))
        curtailment = float(max(0.0, min(1.0, action.curtailment)))

        # ---- Battery dynamics ----
        desired_batt_kw = battery_cmd * BATTERY_MAX_POWER_KW
        battery_power_kw, new_soc = self._apply_battery(
            desired_batt_kw, self._battery_soc
        )

        # ---- Load after curtailment ----
        curtailed_kw = curtailment * flex_load
        load_served_kw = load_kw - curtailed_kw

        # ---- Power balance → grid ----
        # Convention: battery_power_kw > 0 means charging (consuming power)
        net_generation = solar_kw - load_served_kw - battery_power_kw
        # Positive net_generation → export; negative → import
        grid_import_kw = max(0.0, -net_generation)
        grid_export_kw = max(0.0, net_generation)

        # Clamp grid import to distribution limit
        if grid_import_kw > GRID_MAX_IMPORT_KW:
            grid_import_kw = GRID_MAX_IMPORT_KW

        # ---- Reward computation (all components in [0, 1]) ----
        r_cost, r_service, r_solar, r_stability = self._compute_reward(
            solar_kw=solar_kw,
            load_kw=load_kw,
            load_served_kw=load_served_kw,
            grid_import_kw=grid_import_kw,
            grid_export_kw=grid_export_kw,
            curtailed_kw=curtailed_kw,
            battery_power_kw=battery_power_kw,
            new_soc=new_soc,
            price=price,
        )
        # Weighted combination → [0, 1]
        reward = 0.60 * r_cost + 0.20 * r_solar + 0.10 * r_service + 0.10 * r_stability

        # ---- State update ----
        self._battery_soc = new_soc
        self._soc_trajectory.append(new_soc)
        net_cost = grid_import_kw * price * DT - grid_export_kw * price * GRID_EXPORT_PRICE_RATIO * DT
        self._episode_cost += max(0.0, net_cost)
        self._episode_curtailed_kwh += curtailed_kw * DT
        solar_used = min(solar_kw, load_served_kw + max(0.0, battery_power_kw))
        self._episode_solar_used_kwh += solar_used * DT
        self._step += 1

        # Terminal condition: small solar bonus, final reward still in [0, 1]
        terminal_bonus = 0.0
        if self._step >= STEPS_PER_DAY:
            self._done = True
            terminal_bonus = self._terminal_bonus()
            reward = min(1.0, reward + terminal_bonus)

        obs = self._make_obs(
            battery_power_kw=battery_power_kw,
            grid_import_kw=grid_import_kw,
            grid_export_kw=grid_export_kw,
            reward=reward,
            r_cost=r_cost,
            r_curtail=r_service,
            r_renewable=r_solar,
            r_stability=r_stability,
        )

        info = {
            "terminal_bonus": terminal_bonus,
            "solar_kw": solar_kw,
            "load_kw": load_kw,
            "curtailed_kw": curtailed_kw,
            "grid_import_kw": grid_import_kw,
            "grid_export_kw": grid_export_kw,
            "battery_soc": self._battery_soc,
        }
        return obs, reward, self._done, info

    # ------------------------------------------------------------------
    def state(self) -> GridState:
        return GridState(
            episode_id=self._episode_id,
            step=self._step,
            total_steps=STEPS_PER_DAY,
            episode_cost_usd=self._episode_cost,
            done=self._done,
        )

    # ------------------------------------------------------------------
    def episode_summary(self) -> dict:
        """Return stats needed by task graders. Call after done=True."""
        total_solar_kwh = sum(self._solar) * DT
        return {
            "episode_id":             self._episode_id,
            "episode_cost_usd":       self._episode_cost,
            "episode_solar_used_kwh": self._episode_solar_used_kwh,
            "total_solar_kwh":        total_solar_kwh,
            "episode_curtailed_kwh":  self._episode_curtailed_kwh,
            "soc_trajectory":         list(self._soc_trajectory),
            "steps_completed":        self._step,
            "done":                   self._done,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_battery(
        self, desired_kw: float, soc: float
    ) -> Tuple[float, float]:
        """Clamp battery command to SoC and power limits; return (power_kw, new_soc)."""
        actual_kw = desired_kw

        if desired_kw > 0:  # charging
            # Cannot charge above SoC_MAX
            max_charge_kwh = (BATTERY_SOC_MAX - soc) * BATTERY_CAPACITY_KWH
            max_charge_kw = min(BATTERY_MAX_POWER_KW, max_charge_kwh / (DT * BATTERY_EFFICIENCY))
            actual_kw = min(desired_kw, max_charge_kw)
            energy_in = actual_kw * DT * BATTERY_EFFICIENCY
            new_soc = soc + energy_in / BATTERY_CAPACITY_KWH
        elif desired_kw < 0:  # discharging
            # Cannot discharge below SoC_MIN
            max_discharge_kwh = (soc - BATTERY_SOC_MIN) * BATTERY_CAPACITY_KWH
            max_discharge_kw = min(BATTERY_MAX_POWER_KW, max_discharge_kwh * BATTERY_EFFICIENCY / DT)
            actual_kw = max(desired_kw, -max_discharge_kw)
            energy_out = (-actual_kw) * DT / BATTERY_EFFICIENCY
            new_soc = soc - energy_out / BATTERY_CAPACITY_KWH
        else:
            new_soc = soc

        new_soc = max(BATTERY_SOC_MIN, min(BATTERY_SOC_MAX, new_soc))
        return actual_kw, new_soc

    def _compute_reward(
        self,
        solar_kw: float,
        load_kw: float,
        load_served_kw: float,
        grid_import_kw: float,
        grid_export_kw: float,
        curtailed_kw: float,
        battery_power_kw: float,
        new_soc: float,
        price: float,
    ) -> Tuple[float, float, float, float]:
        """
        Return (r_cost, r_service, r_solar, r_stability) all in [0.0, 1.0].

        r_cost      — cost efficiency: 1 = zero net cost, 0 = max possible import at peak
        r_service   — load service:    1 = no curtailment,  0 = all flexible load shed
        r_solar     — solar capture:   1 = all solar used,  0 = all solar exported unused
        r_stability — battery health:  1 = SoC far from limits, 0 = at hard limit
        """
        # 1. Cost efficiency [0, 1]
        net_cost = grid_import_kw * price * DT - grid_export_kw * price * GRID_EXPORT_PRICE_RATIO * DT
        r_cost = max(0.0, min(1.0, 1.0 - net_cost / _MAX_STEP_COST))

        # 2. Load service [0, 1]  (penalises curtailment)
        r_service = min(1.0, load_served_kw / max(load_kw, 0.01))

        # 3. Solar capture [0, 1]
        if solar_kw > 0.5:
            r_solar = max(0.0, min(1.0, (solar_kw - grid_export_kw) / solar_kw))
        else:
            r_solar = 1.0   # no sun available → full marks

        # 4. Battery health [0, 1]
        soc_margin = min(new_soc - BATTERY_SOC_MIN, BATTERY_SOC_MAX - new_soc)
        r_stability = max(0.0, min(1.0, soc_margin / 0.10))

        return r_cost, r_service, r_solar, r_stability

    def _terminal_bonus(self) -> float:
        """End-of-episode solar utilisation bonus, in [0.0, 0.05]."""
        total_solar_kwh = sum(self._solar) * DT
        solar_ratio = (
            self._episode_solar_used_kwh / total_solar_kwh
            if total_solar_kwh > 0.0
            else 1.0
        )
        return 0.05 * max(0.0, min(1.0, solar_ratio))

    def _make_obs(
        self,
        battery_power_kw: float,
        grid_import_kw: float,
        grid_export_kw: float,
        reward: float,
        r_cost: float,
        r_curtail: float,
        r_renewable: float,
        r_stability: float,
    ) -> GridObservation:
        t = min(self._step, STEPS_PER_DAY - 1)
        hour_val = _hour(t)
        solar_val = self._solar[t]
        demand_val = self._load[t]
        flex_val = demand_val * FLEXIBLE_LOAD_FRACTION
        price_val = self._price[t]
        context = _context_text(
            step=self._step,
            hour=hour_val,
            solar_kw=solar_val,
            demand_kw=demand_val,
            flex_kw=flex_val,
            battery_soc=self._battery_soc,
            price=price_val,
            episode_cost=self._episode_cost,
        )
        obs = GridObservation(
            step=self._step,
            hour=hour_val,
            solar_kw=solar_val,
            demand_kw=demand_val,
            flexible_demand_kw=flex_val,
            battery_soc=self._battery_soc,
            battery_power_kw=battery_power_kw,
            electricity_price=price_val,
            grid_import_kw=grid_import_kw,
            grid_export_kw=grid_export_kw,
            episode_cost_usd=self._episode_cost,
            episode_curtailed_kwh=self._episode_curtailed_kwh,
            episode_solar_used_kwh=self._episode_solar_used_kwh,
            done=self._done,
            truncated=False,
            reward=reward,
            reward_energy_cost=r_cost,
            reward_curtailment=r_curtail,
            reward_renewable_bonus=r_renewable,
            reward_stability=r_stability,
            context=context,
        )
        return obs

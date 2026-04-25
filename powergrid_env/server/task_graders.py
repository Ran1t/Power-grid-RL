"""
Three graded tasks for the PowerGrid environment.

Easy   → cost_minimization : minimize electricity cost (single objective)
Medium → solar_arbitrage   : cost + maximize solar self-consumption
Hard   → full_dispatch     : cost + solar + curtailment + active battery use

Calibration anchors (measured empirically on fixed seeds):
  Conservative (do-nothing) agent: cost~$14.6,  solar_frac~0.33, soc_swing~0.0
  Random agent:                    cost~$14-26,  solar_frac~0.60, soc_swing varies
  Target LLM agent:                cost~$8-12,   solar_frac~0.80, soc_swing~0.5+

All grade() scores are in [0.0, 1.0].
"""

from __future__ import annotations

# ── Calibration constants ─────────────────────────────────────────────────────

# Cost ($/day)
_WORST_COST = 30.0   # random agent upper bound → score 0.0
_BEST_COST  = 8.0    # near-optimal battery arbitrage → score 1.0

# Solar utilization (fraction of total available solar captured)
_SOLAR_FLOOR = 0.30  # conservative agent: direct solar→load matching (free, no battery)
_SOLAR_CEIL  = 0.95  # target: capture nearly all surplus via battery charging

# Curtailment
_CURTAIL_THRESHOLD_KWH = 5.0  # 0 curtailed = 1.0, 5+ kWh curtailed = 0.0

# SoC swing (max - min over episode) — rewards active battery trading
_SOC_SWING_FULL = 0.50  # full useful cycle: e.g. charge from 25% → 75% (swing=0.50)


# ── Component scorers ─────────────────────────────────────────────────────────

def _cost_score(episode_cost_usd: float) -> float:
    """
    0.0 = random agent (~$30)   |   1.0 = near-optimal (~$8)
    Conservative do-nothing ~$14.6 → ~0.70
    """
    return max(0.0, min(1.0, (_WORST_COST - episode_cost_usd) / (_WORST_COST - _BEST_COST)))


def _solar_score(solar_used_kwh: float, total_solar_kwh: float) -> float:
    """
    0.0 = only direct solar→load matching (free, no battery needed, ~33%)
    1.0 = ~95% of available solar captured (requires active battery charging at noon)
    Conservative gets ~0.05 here — rewards actual battery management.
    """
    if total_solar_kwh < 0.5:
        return 1.0  # no sun available → full marks (nothing to miss)
    frac = solar_used_kwh / total_solar_kwh
    return max(0.0, min(1.0, (frac - _SOLAR_FLOOR) / (_SOLAR_CEIL - _SOLAR_FLOOR)))


def _curtailment_score(curtailed_kwh: float) -> float:
    """1.0 = no shedding, 0.0 = 5+ kWh of flexible load shed."""
    return max(0.0, 1.0 - curtailed_kwh / _CURTAIL_THRESHOLD_KWH)


def _soc_swing_score(soc_trajectory: list[float]) -> float:
    """
    Measures how actively the agent uses the battery.
    swing = max(SoC) - min(SoC) over the episode.
    0.0 = battery never moved (conservative do-nothing)
    1.0 = battery swung ≥ 50% (e.g. charged from 25% to 75%)
    Rewards deliberate charge-at-noon / discharge-at-peak cycles.
    """
    if not soc_trajectory:
        return 0.0
    swing = max(soc_trajectory) - min(soc_trajectory)
    return min(1.0, swing / _SOC_SWING_FULL)


# ── Task definitions ──────────────────────────────────────────────────────────

TASKS: dict[str, dict] = {
    "cost_minimization": {
        "task_id":    "cost_minimization",
        "difficulty": "easy",
        "seed":       42,
        "description": (
            "Minimize total 24-hour electricity cost. "
            "Use battery arbitrage: charge from cheap off-peak or solar surplus, "
            "discharge during expensive on-peak periods ($0.30/kWh). "
            "Score: 0.0 = random baseline ($30), 1.0 = near-optimal ($8)."
        ),
        "scoring": {"cost": 1.0},
    },
    "solar_arbitrage": {
        "task_id":    "solar_arbitrage",
        "difficulty": "medium",
        "seed":       77,
        "description": (
            "Maximize solar self-consumption AND minimize grid cost. "
            "Agent must charge battery during midday solar surplus and discharge at evening peak. "
            "Score: 60% cost + 40% solar utilization above the passive baseline (0.30 → 0.95)."
        ),
        "scoring": {"cost": 0.6, "solar": 0.4},
    },
    "full_dispatch": {
        "task_id":    "full_dispatch",
        "difficulty": "hard",
        "seed":       13,
        "description": (
            "Full multi-objective dispatch. "
            "Minimize cost, capture solar, avoid curtailment, AND actively cycle the battery "
            "(charge at noon, discharge at peak). A passive do-nothing agent scores ~0.45. "
            "Score: 40% cost + 30% solar + 15% no-curtailment + 15% battery activity."
        ),
        "scoring": {"cost": 0.4, "solar": 0.3, "curtailment": 0.15, "soc_swing": 0.15},
    },
}


# ── Main grader ───────────────────────────────────────────────────────────────

def grade(task_id: str, summary: dict) -> dict:
    """
    Grade a completed (or in-progress) episode on a 0.0–1.0 scale.

    Args:
        task_id: one of TASKS keys
        summary: dict from PowerGridEnvironment.episode_summary()

    Returns dict with:
        score        float [0.0, 1.0]
        components   per-objective breakdown
        episode_stats raw numbers
        calibration  reference anchor values
    """
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASKS)}")

    cost    = summary.get("episode_cost_usd",       999.0)
    solar   = summary.get("episode_solar_used_kwh",   0.0)
    tot_sol = summary.get("total_solar_kwh",           1.0)
    curtail = summary.get("episode_curtailed_kwh",     0.0)
    soc_tr  = summary.get("soc_trajectory",             [])

    cs  = _cost_score(cost)
    ss  = _solar_score(solar, tot_sol)
    cu  = _curtailment_score(curtail)
    sw  = _soc_swing_score(soc_tr)

    weights = TASKS[task_id]["scoring"]
    score = (
          weights.get("cost",        0.0) * cs
        + weights.get("solar",       0.0) * ss
        + weights.get("curtailment", 0.0) * cu
        + weights.get("soc_swing",   0.0) * sw
    )
    score = round(max(0.0, min(1.0, score)), 4)

    return {
        "task_id":    task_id,
        "difficulty": TASKS[task_id]["difficulty"],
        "score":      score,
        "components": {
            "cost_score":        round(cs,  4),
            "solar_score":       round(ss,  4),
            "curtailment_score": round(cu,  4),
            "soc_swing_score":   round(sw,  4),
        },
        "episode_stats": {
            "episode_cost_usd":       round(cost,    4),
            "episode_solar_used_kwh": round(solar,   4),
            "total_solar_kwh":        round(tot_sol, 4),
            "solar_utilization_frac": round(solar / max(tot_sol, 0.1), 4),
            "episode_curtailed_kwh":  round(curtail, 4),
            "soc_swing":              round((max(soc_tr) - min(soc_tr)) if soc_tr else 0.0, 4),
        },
        "calibration": {
            "cost_worst":    _WORST_COST,
            "cost_best":     _BEST_COST,
            "solar_floor":   _SOLAR_FLOOR,
            "solar_ceil":    _SOLAR_CEIL,
        },
    }

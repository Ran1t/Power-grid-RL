"""FastAPI server for the PowerGrid microgrid environment.

OpenEnv-compatible endpoints: POST /reset, POST /step, GET /state
Additional endpoints:         GET /health, GET /tasks, GET /grade/{task_id}, GET /info

Start with:
    uvicorn powergrid_env.server.app:app --host 0.0.0.0 --port 7860

Or via Docker (see Dockerfile in project root).
"""

from __future__ import annotations

import os
import random as _rnd
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

try:
    from ..models import GridAction, GridObservation, GridState, model_to_dict
    from .powergrid_environment import PowerGridEnvironment
    from .task_graders import TASKS, grade as grade_episode
except ImportError:
    from models import GridAction, GridObservation, GridState, model_to_dict
    from powergrid_environment import PowerGridEnvironment
    from task_graders import TASKS, grade as grade_episode


# ---------------------------------------------------------------------------
# Environment configuration from env vars
# ---------------------------------------------------------------------------
_NOISE = float(os.environ.get("PROFILE_NOISE", "0.05"))
_SEED_STR = os.environ.get("SEED", "")
_SEED = int(_SEED_STR) if _SEED_STR else None

_env = PowerGridEnvironment(profile_noise=_NOISE, seed=_SEED)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PowerGrid Microgrid Environment",
    description=(
        "OpenEnv-compatible environment for LLM-based energy management. "
        "Manage a 50 kWh battery and flexible loads over a 24-hour residential microgrid. "
        "Three graded tasks: easy (cost_minimization), medium (solar_arbitrage), hard (full_dispatch)."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------
class ActionRequest(BaseModel):
    battery_action: float = 0.0
    curtailment: float = 0.0

class ResetResponse(BaseModel):
    observation: Dict[str, Any]

class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]

class StateResponse(BaseModel):
    state: Dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _obs_dict(obs: GridObservation) -> Dict[str, Any]:
    return model_to_dict(obs)

def _state_dict(s: GridState) -> Dict[str, Any]:
    return model_to_dict(s)


# ---------------------------------------------------------------------------
# Core OpenEnv endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "environment": "powergrid_env", "version": "1.0.0"}


@app.post("/reset", response_model=ResetResponse)
def reset(task_id: Optional[str] = Query(default=None)) -> ResetResponse:
    """
    Reset the environment for a new episode.
    Pass ?task_id=<id> to use the fixed seed for that task (ensures reproducibility).
    """
    if task_id is not None:
        if task_id not in TASKS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown task_id '{task_id}'. Valid: {list(TASKS)}"
            )
        _rnd.seed(TASKS[task_id]["seed"])
    obs = _env.reset()
    return ResetResponse(observation=_obs_dict(obs))


@app.post("/step", response_model=StepResponse)
def step(body: ActionRequest) -> StepResponse:
    """Take one environment step. battery_action ∈ [-1,1], curtailment ∈ [0,1]."""
    action = GridAction(battery_action=body.battery_action, curtailment=body.curtailment)
    try:
        obs, reward, done, info = _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return StepResponse(observation=_obs_dict(obs), reward=reward, done=done, info=info)


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    """Return lightweight episode metadata (step, cost, done)."""
    return StateResponse(state=_state_dict(_env.state()))


# ---------------------------------------------------------------------------
# Task + grader endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    """List all 3 graded tasks with difficulty and scoring breakdown."""
    return {
        "tasks": [
            {
                "task_id":    t["task_id"],
                "difficulty": t["difficulty"],
                "description": t["description"],
                "scoring":    t["scoring"],
                "reset_url":  f"/reset?task_id={t['task_id']}",
                "grade_url":  f"/grade/{t['task_id']}",
            }
            for t in TASKS.values()
        ]
    }


@app.get("/grade/{task_id}")
def grade(task_id: str) -> Dict[str, Any]:
    """
    Grade the current (or most recently completed) episode on a 0.0–1.0 scale.
    Returns score, component breakdown, and episode statistics.
    """
    if task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Valid: {list(TASKS)}"
        )
    summary = _env.episode_summary()
    return grade_episode(task_id, summary)


# ---------------------------------------------------------------------------
# Info endpoint
# ---------------------------------------------------------------------------

@app.get("/info")
def env_info() -> Dict[str, Any]:
    return {
        "name": "powergrid_env",
        "version": "1.0.0",
        "description": "Microgrid energy management: 96-step 24-hour episode.",
        "action_space": {
            "battery_action": {"type": "float", "range": [-1.0, 1.0],
                               "description": "-1=full discharge, 0=idle, +1=full charge"},
            "curtailment":    {"type": "float", "range": [0.0, 1.0],
                               "description": "Fraction of flexible load to shed"},
        },
        "observation_space": {
            "step":                 "int 0..95",
            "hour":                 "float 0..24",
            "solar_kw":             "float kW",
            "demand_kw":            "float kW",
            "flexible_demand_kw":   "float kW",
            "battery_soc":          "float 0..1",
            "battery_power_kw":     "float kW",
            "electricity_price":    "float $/kWh",
            "grid_import_kw":       "float kW",
            "grid_export_kw":       "float kW",
            "episode_cost_usd":     "float $",
            "done":                 "bool",
            "reward":               "float",
            "context":              "str (human-readable summary for LLM)",
        },
        "tasks": list(TASKS.keys()),
        "episode_length": 96,
        "dt_hours": 0.25,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("powergrid_env.server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()

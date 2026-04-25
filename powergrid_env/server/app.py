"""FastAPI server for the PowerGrid microgrid environment.

Start with:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

Or via Docker (see Dockerfile).
"""

from __future__ import annotations

import os
import inspect
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from ..models import GridAction, GridObservation, GridState, model_to_dict
    from .powergrid_environment import PowerGridEnvironment
except ImportError:
    from models import GridAction, GridObservation, GridState, model_to_dict
    from powergrid_environment import PowerGridEnvironment

# Try using OpenEnv's built-in app factory; fall back to manual FastAPI if missing
_USE_OPENENV_FACTORY = False
try:
    try:
        from openenv_core import create_app
    except ImportError:
        from openenv.core import create_app
    _USE_OPENENV_FACTORY = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Environment configuration from env vars
# ---------------------------------------------------------------------------
_NOISE = float(os.environ.get("PROFILE_NOISE", "0.05"))
_SEED_STR = os.environ.get("SEED", "")
_SEED = int(_SEED_STR) if _SEED_STR else None


# ---------------------------------------------------------------------------
# OpenEnv factory path
# ---------------------------------------------------------------------------
if _USE_OPENENV_FACTORY:
    def _make_env() -> PowerGridEnvironment:
        return PowerGridEnvironment(profile_noise=_NOISE, seed=_SEED)

    _sig = inspect.signature(create_app)
    if "env_factory" in _sig.parameters or "env" in _sig.parameters:
        # Newer OpenEnv: pass a factory callable
        app: FastAPI = create_app(
            env_factory=_make_env,
            action_model=GridAction,
            observation_model=GridObservation,
        )
    else:
        # Older OpenEnv: pass the class
        app = create_app(
            PowerGridEnvironment,
            action_model=GridAction,
            observation_model=GridObservation,
        )

# ---------------------------------------------------------------------------
# Manual FastAPI path (runs without openenv-core installed)
# ---------------------------------------------------------------------------
else:
    app = FastAPI(
        title="PowerGrid Microgrid Environment",
        description=(
            "OpenEnv-compatible environment for LLM-based energy management. "
            "Manage battery storage and load curtailment over a 24-hour horizon."
        ),
        version="1.0.0",
    )

    # One session per process (stateless Spaces deployment)
    _env = PowerGridEnvironment(profile_noise=_NOISE, seed=_SEED)

    # -- Pydantic request/response models for manual FastAPI --
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

    def _obs_to_dict(obs: GridObservation) -> Dict[str, Any]:
        return model_to_dict(obs)

    def _state_to_dict(s: GridState) -> Dict[str, Any]:
        return model_to_dict(s)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "environment": "powergrid_env"}

    @app.post("/reset", response_model=ResetResponse)
    def reset() -> ResetResponse:
        obs = _env.reset()
        return ResetResponse(observation=_obs_to_dict(obs))

    @app.post("/step", response_model=StepResponse)
    def step(body: ActionRequest) -> StepResponse:
        action = GridAction(
            battery_action=body.battery_action,
            curtailment=body.curtailment,
        )
        try:
            obs, reward, done, info = _env.step(action)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return StepResponse(
            observation=_obs_to_dict(obs),
            reward=reward,
            done=done,
            info=info,
        )

    @app.get("/state", response_model=StateResponse)
    def state() -> StateResponse:
        return StateResponse(state=_state_to_dict(_env.state()))

    @app.get("/info")
    def env_info() -> Dict[str, Any]:
        return {
            "name": "powergrid_env",
            "description": (
                "Microgrid energy management: 96-step 24-hour episode. "
                "Actions: battery_action [-1,1], curtailment [0,1]."
            ),
            "action_space": {
                "battery_action": {"type": "float", "range": [-1.0, 1.0]},
                "curtailment": {"type": "float", "range": [0.0, 1.0]},
            },
            "observation_space": {
                "step": "int 0..95",
                "hour": "float 0..24",
                "solar_kw": "float kW",
                "demand_kw": "float kW",
                "flexible_demand_kw": "float kW",
                "battery_soc": "float 0..1",
                "electricity_price": "float $/kWh",
                "grid_import_kw": "float kW",
                "grid_export_kw": "float kW",
                "episode_cost_usd": "float $",
                "done": "bool",
                "reward": "float",
                "context": "str (human-readable summary for LLM)",
            },
            "episode_length": 96,
            "dt_hours": 0.25,
        }


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------
def main() -> None:
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()

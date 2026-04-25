"""
PowerGrid environment client.

Usage (server already running):
    from powergrid_env.client import PowerGridEnv

    env = PowerGridEnv(base_url="http://localhost:8000")
    obs = env.reset()
    obs, reward, done, info = env.step(battery_action=-0.5, curtailment=0.0)

Usage with Docker:
    env = PowerGridEnv.from_docker_image("powergrid-env:latest")
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional, Tuple

try:
    from openenv_core import EnvClient, StepResult
    from openenv_core import State as BaseState
    _HAS_OPENENV = True
except ImportError:
    try:
        from openenv.core import EnvClient, StepResult
        from openenv.core import State as BaseState
        _HAS_OPENENV = True
    except ImportError:
        _HAS_OPENENV = False

try:
    from .models import GridAction, GridObservation, GridState, model_field_names
except ImportError:
    from models import GridAction, GridObservation, GridState, model_field_names


# ---------------------------------------------------------------------------
# OpenEnv EnvClient subclass (preferred when openenv-core is installed)
# ---------------------------------------------------------------------------

if _HAS_OPENENV:
    class PowerGridEnv(EnvClient[GridAction, GridObservation, GridState]):
        """HTTP client for the PowerGrid microgrid environment."""

        def _step_payload(self, action: GridAction) -> Dict[str, Any]:
            return {
                "battery_action": action.battery_action,
                "curtailment": action.curtailment,
            }

        def _parse_result(self, data: Dict[str, Any]) -> "StepResult[GridObservation]":
            obs_data = data.get("observation", data)
            obs_fields = model_field_names(GridObservation)
            obs = GridObservation(**{k: v for k, v in obs_data.items() if k in obs_fields})
            return StepResult(
                observation=obs,
                reward=data.get("reward", 0.0),
                done=data.get("done", False),
                info=data.get("info", {}),
            )

        def _parse_state(self, data: Dict[str, Any]) -> GridState:
            state_data = data.get("state", data)
            state_fields = model_field_names(GridState)
            return GridState(**{k: v for k, v in state_data.items() if k in state_fields})

        def step(
            self,
            battery_action: float = 0.0,
            curtailment: float = 0.0,
        ) -> "StepResult[GridObservation]":
            """Convenience wrapper: accepts kwargs instead of an action object."""
            action = GridAction(
                battery_action=battery_action,
                curtailment=curtailment,
            )
            return super().step(action)

# ---------------------------------------------------------------------------
# Lightweight HTTP client (fallback when openenv-core is NOT installed)
# ---------------------------------------------------------------------------

else:
    import json
    try:
        import requests
    except ImportError:
        requests = None  # type: ignore[assignment]

    class PowerGridEnv:  # type: ignore[no-redef]
        """
        Minimal HTTP client for the PowerGrid server.
        Install openenv-core for the full-featured EnvClient.
        """

        def __init__(self, base_url: str = "http://localhost:8000") -> None:
            if requests is None:
                raise ImportError(
                    "Install 'requests': pip install requests"
                )
            self.base_url = base_url.rstrip("/")

        # -- factory --
        @classmethod
        def from_docker_image(cls, image: str, port: int = 8000) -> "PowerGridEnv":
            import subprocess, time
            subprocess.Popen(
                ["docker", "run", "-d", "-p", f"{port}:8000", image],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(3)
            return cls(base_url=f"http://localhost:{port}")

        # -- core API --
        def reset(self) -> GridObservation:
            r = requests.post(f"{self.base_url}/reset")
            r.raise_for_status()
            data = r.json()
            obs_data = data.get("observation", data)
            return self._parse_obs(obs_data)

        def step(
            self,
            battery_action: float = 0.0,
            curtailment: float = 0.0,
        ) -> Tuple[GridObservation, float, bool, Dict[str, Any]]:
            payload = {"battery_action": battery_action, "curtailment": curtailment}
            r = requests.post(f"{self.base_url}/step", json=payload)
            r.raise_for_status()
            data = r.json()
            obs = self._parse_obs(data.get("observation", {}))
            return obs, data.get("reward", 0.0), data.get("done", False), data.get("info", {})

        def state(self) -> GridState:
            r = requests.get(f"{self.base_url}/state")
            r.raise_for_status()
            data = r.json().get("state", r.json())
            return GridState(**{
                k: v for k, v in data.items()
                if k in GridState.__dataclass_fields__
            })

        def info(self) -> Dict[str, Any]:
            r = requests.get(f"{self.base_url}/info")
            r.raise_for_status()
            return r.json()

        @staticmethod
        def _parse_obs(data: Dict[str, Any]) -> GridObservation:
            fields = model_field_names(GridObservation)
            return GridObservation(**{k: v for k, v in data.items() if k in fields})

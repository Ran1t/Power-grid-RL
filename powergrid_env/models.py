"""
PowerGrid environment data models.

Supports both openenv-core (Pydantic BaseModel) and plain-Python (dataclass)
base classes automatically. Do NOT add @dataclass to classes that inherit
from a Pydantic BaseModel — Pydantic handles field creation from annotations.
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Base class detection
# ---------------------------------------------------------------------------

_PYDANTIC_PARENT = False

try:
    from openenv_core import Action, Observation, State

    try:
        from pydantic import BaseModel as _BM
        _PYDANTIC_PARENT = issubclass(Observation, _BM)
    except (ImportError, TypeError):
        pass

except ImportError:
    try:
        from openenv.core import Action, Observation, State

        try:
            from pydantic import BaseModel as _BM
            _PYDANTIC_PARENT = issubclass(Observation, _BM)
        except (ImportError, TypeError):
            pass

    except ImportError:
        # Fallback: define plain-Python dataclass stubs
        from dataclasses import dataclass as _dc

        @_dc
        class Action:  # type: ignore[no-redef]
            pass

        @_dc
        class Observation:  # type: ignore[no-redef]
            pass

        @_dc
        class State:  # type: ignore[no-redef]
            pass


# ---------------------------------------------------------------------------
# Conditional dataclass decorator
# ---------------------------------------------------------------------------
# When openenv_core's base is Pydantic, Pydantic handles field creation from
# annotated class variables automatically — @dataclass is WRONG.
# When using the fallback plain-Python stubs, we need @dataclass.

def _maybe_dc(cls):
    """Apply @dataclass only when parents are plain Python (not Pydantic)."""
    if not _PYDANTIC_PARENT:
        from dataclasses import dataclass
        return dataclass(cls)
    return cls


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

@_maybe_dc
class GridAction(Action):
    """Agent controls battery charge/discharge and flexible load curtailment."""

    battery_action: float = 0.0
    curtailment: float = 0.0

    if not _PYDANTIC_PARENT:
        def __post_init__(self) -> None:
            self.battery_action = float(max(-1.0, min(1.0, self.battery_action)))
            self.curtailment = float(max(0.0, min(1.0, self.curtailment)))


@_maybe_dc
class GridObservation(Observation):
    """Full microgrid state visible to the agent each timestep."""

    # --- Time ---
    step: int = 0
    hour: float = 0.0

    # --- Generation ---
    solar_kw: float = 0.0

    # --- Demand ---
    demand_kw: float = 0.0
    flexible_demand_kw: float = 0.0

    # --- Battery ---
    battery_soc: float = 0.5
    battery_power_kw: float = 0.0

    # --- Grid ---
    electricity_price: float = 0.15
    grid_import_kw: float = 0.0
    grid_export_kw: float = 0.0

    # --- Episode metrics ---
    episode_cost_usd: float = 0.0
    episode_curtailed_kwh: float = 0.0
    episode_solar_used_kwh: float = 0.0

    # --- Terminal flag ---
    done: bool = False
    truncated: bool = False

    # --- Reward ---
    reward: float = 0.0
    reward_energy_cost: float = 0.0
    reward_curtailment: float = 0.0
    reward_renewable_bonus: float = 0.0
    reward_stability: float = 0.0

    # --- Context for LLM ---
    context: str = ""


@_maybe_dc
class GridState(State):
    """Lightweight server-side episode metadata."""

    episode_id: str = ""
    step: int = 0
    total_steps: int = 96
    episode_cost_usd: float = 0.0
    done: bool = False


# ---------------------------------------------------------------------------
# Serialization helpers (work for both Pydantic and dataclass models)
# ---------------------------------------------------------------------------

def model_to_dict(obj) -> dict:
    """Convert a model to a plain dict, regardless of whether it is Pydantic or dataclass."""
    # Pydantic V2
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    # Pydantic V1
    if hasattr(obj, "dict"):
        return obj.dict()
    # dataclass
    import dataclasses
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    # plain class / fallback
    return obj.__dict__.copy()


def model_field_names(cls) -> set:
    """Return the set of field names for a model class."""
    # Pydantic V2
    if hasattr(cls, "model_fields"):
        return set(cls.model_fields.keys())
    # Pydantic V1
    if hasattr(cls, "__fields__"):
        return set(cls.__fields__.keys())
    # dataclass
    import dataclasses
    try:
        return {f.name for f in dataclasses.fields(cls)}
    except TypeError:
        pass
    # plain class annotations
    return set(getattr(cls, "__annotations__", {}).keys())

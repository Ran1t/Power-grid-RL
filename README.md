---
title: PowerGrid Microgrid Balancing
emoji: ⚡
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
license: mit
---

# PowerGrid Microgrid Balancing — OpenEnv Environment

**Hackathon Theme:** World Modeling — Professional Tasks (Theme 3.1)
**Framework:** [OpenEnv](https://github.com/meta-pytorch/OpenEnv) (latest release)
**Training:** GRPO via [Unsloth](https://github.com/unslothai/unsloth) + HuggingFace TRL

> An LLM agent manages a 50 kWh residential battery and flexible loads over a 24-hour
> period to minimize electricity costs, maximize solar utilization, and maintain grid stability.
> 96 decision steps × 15-minute intervals. Dense reward every step.

---

## The Problem

Real residential microgrids (solar + battery + grid) waste **15–30% of potential savings**
because battery scheduling is done with rigid, time-invariant rules that don't adapt to:
- Dynamic time-of-use (TOU) pricing
- Variable solar generation (clouds, seasons)
- Changing load patterns

An LLM agent can reason *why* a decision makes sense ("it's peak pricing window, battery at
65%, discharge now") rather than memorizing a fixed rule — making it adaptable.

## Environment Design

```
Microgrid topology:
  [Solar PV 20 kW peak] ─┐
  [Grid connection]      ─┼── [Battery 50 kWh / 25 kW] ── [Residential Load 3–23 kW]
  [Agent controller]     ─┘
```

### Episode
- **Horizon:** 96 steps × 15 min = 24 hours
- **New profile every episode** (Gaussian noise on solar/load curves)
- **Starting SoC:** randomized 20–80%

### Action Space
| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `battery_action` | float | [-1, 1] | -1 = full discharge, 0 = idle, +1 = full charge |
| `curtailment` | float | [0, 1] | Fraction of flexible load to shed |

### Observation Space
| Field | Unit | Description |
|-------|------|-------------|
| `step` | int | Timestep 0–95 |
| `hour` | float | Hour of day 0–24 |
| `solar_kw` | kW | PV generation |
| `demand_kw` | kW | Total load |
| `flexible_demand_kw` | kW | Deferrable load (30%) |
| `battery_soc` | 0–1 | State of charge |
| `electricity_price` | $/kWh | Time-of-use tariff |
| `grid_import_kw` | kW | Power bought from grid |
| `episode_cost_usd` | $ | Cumulative cost |
| `context` | str | **Human-readable summary for LLM** |

### Reward Function
```
r_step = -energy_cost + renewable_bonus - curtailment_penalty - stability_penalty

energy_cost       = grid_import × price × 0.25 h
renewable_bonus   = 0.05 × solar_fraction × solar_kw × 0.25 h
curtailment_pen   = 0.50 $/kWh × curtailed_load × 0.25 h
stability_pen     = -0.5 when battery SoC near hard limits (10% or 95%)

r_terminal = min(5, 5 × solar_ratio - 0.1 × total_cost)   # end-of-day bonus
```

### Time-of-Use Pricing
| Period | Hours | Price |
|--------|-------|-------|
| OFF-PEAK | 21:00–07:00 | $0.08/kWh |
| MID-PEAK | 11:00–17:00 | $0.15/kWh |
| ON-PEAK | 07:00–11:00, 17:00–21:00 | $0.30/kWh |

---

## Quick Start

### Run the server locally
```bash
pip install fastapi uvicorn numpy pydantic
cd powergrid_env
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Interact via Python client
```python
from powergrid_env.client import PowerGridEnv

env = PowerGridEnv(base_url="http://localhost:8000")
obs = env.reset()

for step in range(96):
    print(obs['context'])
    # Discharge at peak, charge at solar noon
    battery_action = -0.8 if obs['electricity_price'] >= 0.30 else 0.5
    obs, reward, done, info = env.step(battery_action=battery_action, curtailment=0.0)
    if done:
        break
```

### Run with Docker
```bash
cd powergrid_env
docker build -f server/Dockerfile -t powergrid-env .
docker run -p 8000:8000 powergrid-env
```

---

## File Structure

```
powergrid_env/
├── __init__.py
├── models.py                        # GridAction, GridObservation, GridState
├── client.py                        # PowerGridEnv (EnvClient subclass)
├── openenv.yaml                     # OpenEnv manifest
├── pyproject.toml                   # Package dependencies
└── server/
    ├── __init__.py
    ├── powergrid_environment.py     # Physics simulation + reward logic
    ├── app.py                       # FastAPI server (OpenEnv-compatible)
    ├── requirements.txt
    └── Dockerfile

training/
└── train_powergrid.ipynb            # GRPO training (Colab-ready)

outputs/
├── logs/
└── evals/
```

---

## Training

Open `training/train_powergrid.ipynb` in Google Colab (T4 GPU, free tier).

**Model:** `Qwen/Qwen2.5-1.5B-Instruct` fine-tuned with GRPO  
**Budget:** ~$5–10 of HuggingFace compute credit for 3 epochs on A100

### What GRPO trains
The model receives the grid state as a structured prompt and must output a JSON action.
Two reward functions guide training:
1. **Environment reward** — scored against the physics simulator
2. **Format reward** — +1.5 for valid JSON with correct keys/ranges

### Expected results after training
| Agent | Episode Reward | Energy Cost |
|-------|---------------|-------------|
| Random | ~-25 | ~$8.50 |
| Rule-based | ~-12 | ~$5.20 |
| **GRPO LLM** | **~-8** | **~$4.10** |

---

## HuggingFace Space

**Environment Space:** `https://huggingface.co/spaces/Ran1t/powergrid-env`  
**Trained Model:** `https://huggingface.co/Ran1t/powergrid-grpo-qwen2.5-1.5b`

---

## Judging Criteria Alignment

| Criterion | Implementation |
|-----------|---------------|
| **Environment Innovation (40%)** | Novel LLM-controlled energy management; real-world domain underexplored in LLM RL training; physics-based simulation with dense rewards |
| **Storytelling (30%)** | Clear problem (battery arbitrage), mechanism (TOU pricing), and outcome (cost reduction) |
| **Reward Improvement (20%)** | Dense per-step reward; terminal bonus; 3-way comparison (random / rule / LLM) with plots |
| **Training Pipeline (10%)** | GRPO via Unsloth + TRL; composable reward functions; reproducible Colab notebook |

---

## Why Power Grid? Why LLMs?

1. **Underexplored domain** — Most RL benchmarks are games or locomotion. Real energy management is high-stakes, complex, and deployable.
2. **Long-horizon reasoning** — Optimal battery dispatch requires reasoning 6–12 hours ahead (charge now for peak later).
3. **Natural language alignment** — Utility operators already reason in text ("peak pricing window", "solar forecast"). LLMs bridge the gap between human expertise and automated control.
4. **Immediate real-world impact** — $4–8/day savings per household × millions of homes = significant grid decarbonization.

---

## References

- [PowerGridworld (NREL)](https://github.com/NatLabRockies/PowerGridworld) — multi-agent power systems framework
- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) — environment framework
- [TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer) — training algorithm
- [Unsloth](https://github.com/unslothai/unsloth) — efficient LLM fine-tuning

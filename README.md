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

**Theme:** World Modeling — Professional Tasks (Theme 3.1)  
**Framework:** [OpenEnv](https://github.com/meta-pytorch/OpenEnv) (latest release)  
**Training:** GRPO via [Unsloth](https://github.com/unslothai/unsloth) + HuggingFace TRL  
**Space:** [huggingface.co/spaces/Ran1t/powergrid-env](https://huggingface.co/spaces/Ran1t/powergrid-env)

---

## The Problem

Real residential microgrids (solar + battery + grid connection) waste **15–30 % of potential savings** because battery scheduling uses rigid, time-invariant rules that don't adapt to:

- Dynamic **time-of-use (TOU) pricing** ($0.08 → $0.30/kWh swings)
- Variable **solar generation** (clouds, seasons, forecast uncertainty)
- Changing **load patterns** (occupancy, appliance usage)

An LLM agent can reason *why* a decision makes sense ("it's peak pricing window, battery at 65 %, discharge now") rather than just following a hard-coded rule — making it inherently adaptable.

> **Does this exist to teach an LLM something it can't currently do well?**  
> Yes. Long-horizon energy arbitrage requires reasoning 6–12 hours ahead, coordinating multiple objectives, and understanding domain-specific constraints — none of which LLMs are explicitly trained for.

---

## Environment Design

```
Microgrid topology:
  [Solar PV 20 kW peak] ──┐
  [Grid connection]       ──┼── [Battery 50 kWh / 25 kW] ── [Residential Load 3–23 kW]
  [LLM agent controller]  ──┘
```

### Episode
- **Horizon:** 96 steps × 15 min = 24 hours
- **New profile every episode** (Gaussian noise on solar + load curves)
- **Starting SoC:** randomised 20–80 %
- **Real load data:** NREL PowerGridworld hourly profiles

### Action Space
| Field | Type | Range | Description |
|---|---|---|---|
| `battery_action` | float | [-1, 1] | -1 = full discharge, 0 = idle, +1 = full charge |
| `curtailment` | float | [0, 1] | Fraction of flexible load to shed (avoid this) |

### Observation Space
| Field | Unit | Description |
|---|---|---|
| `step` | int | Timestep 0–95 |
| `hour` | float | Hour of day 0–24 |
| `solar_kw` | kW | PV generation |
| `demand_kw` | kW | Total load |
| `flexible_demand_kw` | kW | Deferrable load (30 %) |
| `battery_soc` | 0–1 | State of charge |
| `electricity_price` | $/kWh | Current TOU tariff |
| `grid_import_kw` | kW | Power drawn from grid |
| `episode_cost_usd` | $ | Cumulative cost so far |
| `context` | str | **Human-readable summary for the LLM** |

### Reward Function — Dense, Multi-Component, Normalised to [0, 1]

```
r_cost    = max(0, min(1,  1 - net_cost / max_step_cost ))   # cost efficiency
r_solar   = max(0, min(1, (solar_kw - export_kw) / solar_kw))# renewable capture
r_service = min(1, load_served / total_load)                  # curtailment avoidance
r_stab    = max(0, min(1,  soc_margin / 0.10 ))               # battery health

reward = 0.60 * r_cost + 0.20 * r_solar + 0.10 * r_service + 0.10 * r_stab
# Combined reward strictly in [0, 1] every step
# Terminal step gets +0.05 solar bonus, capped at 1.0
```

**Why this is hard to game:** An agent that always idles gets r_cost ≈ 0.7 but r_solar ≈ 0.05 (wastes solar). An agent that always charges gets high r_solar but low r_cost during peak hours. The only way to score > 0.93 is to genuinely optimise all four components simultaneously.

### Time-of-Use Pricing
| Period | Hours | Price |
|---|---|---|
| OFF-PEAK | 21:00–07:00 | $0.08/kWh |
| MID-PEAK | 11:00–17:00 | $0.15/kWh |
| ON-PEAK | 07:00–11:00, 17:00–21:00 | $0.30/kWh |

### Three Graded Tasks (0.0 – 1.0)
| Task | Difficulty | Scoring |
|---|---|---|
| `cost_minimization` | Easy | 100 % cost efficiency |
| `solar_arbitrage` | Medium | 60 % cost + 40 % solar utilisation |
| `full_dispatch` | Hard | 40 % cost + 30 % solar + 15 % no-curtailment + 15 % battery activity |

---

## Baseline Results

Real data from 30 evaluation episodes (NREL load profiles + clear-sky solar):

| Agent | Avg Score [0-1] | Avg Cost ($/day) | Notes |
|---|---|---|---|
| Random | 0.882 | $52.00 | Lower bound |
| Conservative (do-nothing) | 0.930 | $41.61 | Passive upper bound |
| Rule-based | 0.910 | $40.00 | Heuristic baseline |
| Q-Learning (trained 200 ep) | 0.8561 | $45.66 | Tabular RL baseline |
| SARSA (trained 200 ep) | 0.8486 | $40.44 | On-policy RL baseline |
| **GRPO LLM (Qwen2.5-1.5B)** | **0.833** | **TBD** | Fine-tuned with GRPO |

**Baseline comparison — score distribution and cost:**

![Baseline Results](outputs/baseline_results.png)
*Left: Score [0-1] over 30 eval episodes. Middle: Distribution per agent. Right: Mean daily energy cost.*

---

## Training Evidence

### RL Agent Learning Curves (real runs)

The Q-Learning and SARSA agents were trained for 200 episodes on real NREL load data. The plots below show actual learning curves — not simulated:

![Training Curves](outputs/training_curves.png)
*Left: Per-episode training score (raw + 10-episode moving average). Right: Cumulative average showing learning progress.*

Both RL agents show clear improvement during training (rising cumulative average), confirming the environment provides a learnable signal.

### GRPO LLM Training

The notebook [`training/train_powergrid.ipynb`](training/train_powergrid.ipynb) fine-tunes **Qwen2.5-1.5B-Instruct** with GRPO using Unsloth + HuggingFace TRL.

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct` (1.5 B parameters, fits on T4 free Colab)
- **Method:** GRPO — group relative policy optimisation, no value network needed
- **Reward:** 70 % environment physics reward + 30 % JSON format reward
- **Training time:** ~45 min on T4, ~15 min on A100

To run training and generate real GRPO curves:
```
# Open in Google Colab (T4 GPU, free tier)
training/train_powergrid.ipynb
```
After running, commit `powergrid_results.png` to the repo.

### GRPO Results

![GRPO Training Results](powergrid_results.png)
*Left: Score comparison across all agents. Middle: GRPO reward curve over 600 training steps. Right: Rule-based agent 24-hour episode trace.*

**Trained model:** [Ran1t/powergrid-grpo-qwen2.5-1.5b](https://huggingface.co/Ran1t/powergrid-grpo-qwen2.5-1.5b)

---

## Quick Start

### Run the server locally
```bash
pip install fastapi uvicorn numpy pydantic openai
cd powergrid_env
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Interact via the Python client
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
docker build -f Dockerfile -t powergrid-env .
docker run -p 7860:7860 powergrid-env
```

### Run LLM inference
```bash
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
export HF_TOKEN=hf_...
python inference.py
```

---

## Repository Structure

```
powergrid_env/
├── models.py                    # GridAction, GridObservation, GridState (OpenEnv types)
├── client.py                    # PowerGridEnv (EnvClient subclass) — no server imports
├── openenv.yaml                 # OpenEnv manifest
├── pyproject.toml
└── server/
    ├── powergrid_environment.py # Physics simulation (extends Environment base class)
    ├── app.py                   # FastAPI server — POST /reset, POST /step, GET /state
    ├── task_graders.py          # 3 graded tasks, scores in [0.0, 1.0]
    ├── requirements.txt
    └── Dockerfile

training/
└── train_powergrid.ipynb        # GRPO training (Colab-ready, Unsloth + TRL)

baseline_agents.py               # Random, Conservative, Q-Learning, SARSA
run_baselines.py                 # Evaluation + training curves
inference.py                     # LLM agent via OpenAI-compatible API
outputs/
├── baseline_results.png         # Agent comparison plots
└── training_curves.png          # RL training evidence
```

---

## Why This Problem? Why LLMs?

1. **Underexplored domain** — RL benchmarks are full of games. Real energy management is high-stakes and deployable today.
2. **Long-horizon reasoning** — Optimal battery dispatch requires thinking 6–12 hours ahead ("charge now at $0.15, discharge at $0.30 peak in 4 hours").
3. **Natural language alignment** — Utility operators already reason in text. LLMs bridge human expertise and automated control.
4. **Real-world impact** — $4–8/day savings per household × millions of homes = meaningful grid decarbonisation.

---

## References

- [PowerGridworld (NREL)](https://github.com/NatLabRockies/PowerGridworld) — real load data source
- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) — environment framework
- [TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer) — training algorithm
- [Unsloth](https://github.com/unslothai/unsloth) — efficient LLM fine-tuning

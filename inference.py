"""
PowerGrid inference script — OpenEnv hackathon submission.

Runs 3 graded tasks (easy / medium / hard) using an LLM agent via OpenAI-compatible API.
Emits structured [START], [STEP], [END] logs to stdout.

Environment variables (required):
  API_BASE_URL   OpenAI-compatible endpoint  (e.g. https://api.openai.com/v1)
  MODEL_NAME     Model identifier            (e.g. Qwen/Qwen2.5-1.5B-Instruct)
  HF_TOKEN       API key / HuggingFace token

Optional:
  ENV_BASE_URL   PowerGrid server URL (default: inline environment, no server needed)
  MAX_STEPS      Steps per episode (default: 96)

Usage:
  python inference.py
  python inference.py --task cost_minimization    # single task
  python inference.py --steps 30                  # faster eval
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime, timezone


# ── Environment variables ─────────────────────────────────────────────────────

def _require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        print(f"[ERROR] Environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val


def _load_dotenv():
    path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-1.5B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "")   # empty = use inline env


# ── Inline microgrid environment (no server needed) ───────────────────────────

BATTERY_CAPACITY_KWH = 50.0
BATTERY_MAX_POWER_KW = 25.0
BATTERY_EFFICIENCY   = 0.95
BATTERY_SOC_MIN      = 0.10
BATTERY_SOC_MAX      = 0.95
SOLAR_PEAK_KW        = 20.0
FLEX_FRAC            = 0.30
CURTAIL_PENALTY      = 0.50
EXPORT_RATIO         = 0.40
STEPS                = 96
DT                   = 0.25


def _hour(t):     return t * DT
def _solar(t):
    h = _hour(t)
    if h < 6 or h > 18: return 0.0
    return max(0.0, SOLAR_PEAK_KW * math.exp(-((h - 12) ** 2) / 8) * (1 + 0.05 * random.gauss(0, 1)))
def _load(t):
    h = _hour(t)
    return max(0.5, (3 + 8*math.exp(-((h-8)**2)/2) + 12*math.exp(-((h-19)**2)/2)) * (1 + 0.05*random.gauss(0, 0.5)))
def _price(t):
    h = _hour(t)
    if (7 <= h < 11) or (17 <= h < 21): return 0.30
    if 11 <= h < 17: return 0.15
    return 0.08


class _Obs:
    __slots__ = ["step","hour","solar_kw","demand_kw","battery_soc","price",
                 "grid_import_kw","episode_cost","done","reward","context",
                 "_sol","_lod","_prc","_soc","_curtailed","_solar_used"]


class InlineMicrogridEnv:
    def reset(self, seed=None):
        if seed is not None: random.seed(seed)
        self.t = 0; self.soc = random.uniform(0.2, 0.8)
        self.cost = 0.0; self.done = False
        self.curtailed_kwh = 0.0; self.solar_used_kwh = 0.0
        self._s = [_solar(t) for t in range(STEPS)]
        self._l = [_load(t)  for t in range(STEPS)]
        self._p = [_price(t) for t in range(STEPS)]
        self._soc_traj = []
        return self._make_obs(0, 0, 0)

    def step(self, batt_cmd, curtail):
        batt_cmd = max(-1.0, min(1.0, float(batt_cmd)))
        curtail  = max(0.0,  min(1.0, float(curtail)))
        t = self.t
        s, l, p = self._s[t], self._l[t], self._p[t]
        desired = batt_cmd * BATTERY_MAX_POWER_KW
        batt_kw, self.soc = self._battery(desired, self.soc)
        self._soc_traj.append(self.soc)
        curtailed = curtail * l * FLEX_FRAC
        served = l - curtailed
        net = s - served - batt_kw
        imp = max(0.0, -net); exp_ = max(0.0, net)
        _msc    = 50.0 * 0.30 * DT
        net_cost = imp*p*DT - exp_*p*EXPORT_RATIO*DT
        r_cost  = max(0.0, min(1.0, 1.0 - net_cost / _msc))
        r_solar = max(0.0, min(1.0, (s - exp_) / s)) if s > 0.5 else 1.0
        r_svc   = min(1.0, served / max(l, 0.01))
        soc_m   = min(self.soc-BATTERY_SOC_MIN, BATTERY_SOC_MAX-self.soc)
        r_stab  = max(0.0, min(1.0, soc_m / 0.10))
        r = 0.60*r_cost + 0.20*r_solar + 0.10*r_svc + 0.10*r_stab

        self.cost += max(0.0, net_cost)
        self.curtailed_kwh += curtailed * DT
        solar_used = min(s, served + max(0.0, batt_kw))
        self.solar_used_kwh += solar_used * DT
        self.t += 1
        if self.t >= STEPS:
            self.done = True
            r = min(1.0, r + 0.05)
        return self._make_obs(imp, r, batt_kw), r, self.done, {}

    def _battery(self, desired, soc):
        if desired > 0:
            mk = min(BATTERY_MAX_POWER_KW, (BATTERY_SOC_MAX-soc)*BATTERY_CAPACITY_KWH/(DT*BATTERY_EFFICIENCY))
            a = min(desired, mk); s2 = soc + a*DT*BATTERY_EFFICIENCY/BATTERY_CAPACITY_KWH
        elif desired < 0:
            mk = min(BATTERY_MAX_POWER_KW, (soc-BATTERY_SOC_MIN)*BATTERY_CAPACITY_KWH*BATTERY_EFFICIENCY/DT)
            a = max(desired, -mk); s2 = soc - (-a)*DT/BATTERY_EFFICIENCY/BATTERY_CAPACITY_KWH
        else:
            a, s2 = 0.0, soc
        return a, max(BATTERY_SOC_MIN, min(BATTERY_SOC_MAX, s2))

    def _make_obs(self, imp, reward, batt_kw):
        t = min(self.t, STEPS-1)
        h = _hour(t); hi, mi = int(h), int((h-int(h))*60)
        p = self._p[t]
        pl = "ON-PEAK" if p >= 0.30 else "MID-PEAK" if p >= 0.15 else "OFF-PEAK"
        o = _Obs()
        o.step = self.t; o.hour = h; o.solar_kw = self._s[t]; o.demand_kw = self._l[t]
        o.battery_soc = self.soc; o.price = p; o.grid_import_kw = imp
        o.episode_cost = self.cost; o.done = self.done; o.reward = reward
        o.context = (f"Time {hi:02d}:{mi:02d} | Step {self.t}/96 | "
                     f"Solar {self._s[t]:.1f} kW | Demand {self._l[t]:.1f} kW | "
                     f"SoC {self.soc*100:.0f}% | {p:.2f} $/kWh ({pl}) | Cost ${self.cost:.2f}")
        return o

    def episode_summary(self):
        total_solar = sum(self._s) * DT
        return {
            "episode_cost_usd":       self.cost,
            "episode_solar_used_kwh": self.solar_used_kwh,
            "total_solar_kwh":        total_solar,
            "episode_curtailed_kwh":  self.curtailed_kwh,
            "soc_trajectory":         list(self._soc_traj),
        }


# ── Task definitions ──────────────────────────────────────────────────────────

TASKS = [
    {"task_id": "cost_minimization", "difficulty": "easy",   "seed": 42},
    {"task_id": "solar_arbitrage",   "difficulty": "medium",  "seed": 77},
    {"task_id": "full_dispatch",     "difficulty": "hard",    "seed": 13},
]

_WORST_COST  = 30.0
_BEST_COST   = 8.0
_SOLAR_FLOOR = 0.30
_SOLAR_CEIL  = 0.95
_SOC_SWING_FULL = 0.50


def _grade(task_id: str, summary: dict) -> float:
    cost    = summary["episode_cost_usd"]
    solar   = summary["episode_solar_used_kwh"]
    tot_s   = max(summary["total_solar_kwh"], 0.1)
    curtail = summary["episode_curtailed_kwh"]
    soc_tr  = summary["soc_trajectory"]

    cs  = max(0.0, min(1.0, (_WORST_COST - cost) / (_WORST_COST - _BEST_COST)))
    ss  = max(0.0, min(1.0, (solar/tot_s - _SOLAR_FLOOR) / (_SOLAR_CEIL - _SOLAR_FLOOR)))
    cu  = max(0.0, 1.0 - curtail / 5.0)
    sw  = min(1.0, (max(soc_tr) - min(soc_tr)) / _SOC_SWING_FULL) if soc_tr else 0.0

    if task_id == "cost_minimization": return round(cs, 4)
    if task_id == "solar_arbitrage":   return round(0.6*cs + 0.4*ss, 4)
    if task_id == "full_dispatch":     return round(0.4*cs + 0.3*ss + 0.15*cu + 0.15*sw, 4)
    return 0.0


# ── LLM agent via OpenAI-compatible client ────────────────────────────────────

SYSTEM_PROMPT = """You are an expert power grid operator managing a residential microgrid.
Minimize electricity costs over 24 hours by controlling a 50 kWh battery.

Prices: OFF-PEAK $0.08 (21:00-07:00), MID-PEAK $0.15, ON-PEAK $0.30 (07-11, 17-21h)
Solar PV: up to 20 kW peak at noon.

Respond ONLY with a JSON object:
{"battery_action": <float -1.0 to 1.0>, "curtailment": <float 0.0 to 1.0>}
battery_action: -1=full discharge, 0=idle, +1=full charge. curtailment: 0=no shedding."""


def _make_prompt(obs) -> str:
    steps_left = 96 - obs.step
    if obs.price >= 0.30 and obs.battery_soc > 0.3:
        hint = "DISCHARGE — peak price, sell battery power now"
    elif obs.solar_kw > 8 and obs.price <= 0.15:
        hint = "CHARGE — solar surplus available, store it"
    elif obs.price < 0.10 and obs.battery_soc < 0.6:
        hint = "CHARGE — cheap off-peak, top up battery"
    else:
        hint = "HOLD — no strong signal"
    return f"{obs.context}\nSteps remaining: {steps_left}/96\nHint: {hint}\n\nJSON action:"


def _parse(text: str):
    try:
        m = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if m:
            d = json.loads(m.group())
            return (max(-1.0, min(1.0, float(d.get("battery_action", 0.0)))),
                    max(0.0,  min(1.0, float(d.get("curtailment", 0.0)))))
    except Exception:
        pass
    bm = re.search(r'battery_action["\s:]+([-\d.]+)', text)
    cm = re.search(r'curtailment["\s:]+([\d.]+)', text)
    return (max(-1.0, min(1.0, float(bm.group(1)))) if bm else 0.0,
            max(0.0,  min(1.0, float(cm.group(1)))) if cm else 0.0)


def _llm_act(client, model: str, obs) -> tuple[float, float]:
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": _make_prompt(obs)},
                ],
                max_tokens=64,
                temperature=0.1,
            )
            return _parse(resp.choices[0].message.content.strip())
        except Exception:
            time.sleep(1)
    return 0.0, 0.0


# ── Structured logging ────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_start(task: dict):
    print(json.dumps({
        "event":      "[START]",
        "task_id":    task["task_id"],
        "difficulty": task["difficulty"],
        "seed":       task["seed"],
        "model":      MODEL_NAME,
        "timestamp":  _now(),
    }), flush=True)


def log_step(step: int, obs, action: tuple, reward: float, score_so_far: float):
    print(json.dumps({
        "event":          "[STEP]",
        "step":           step,
        "hour":           round(obs.hour, 2),
        "solar_kw":       round(obs.solar_kw, 2),
        "demand_kw":      round(obs.demand_kw, 2),
        "battery_soc":    round(obs.battery_soc, 3),
        "price":          obs.price,
        "grid_import_kw": round(obs.grid_import_kw, 2),
        "action": {
            "battery_action": round(action[0], 3),
            "curtailment":    round(action[1], 3),
        },
        "reward":         round(reward, 4),
        "episode_cost":   round(obs.episode_cost, 4),
        "done":           obs.done,
    }), flush=True)


def log_end(task: dict, score: float, summary: dict, elapsed: float):
    print(json.dumps({
        "event":              "[END]",
        "task_id":            task["task_id"],
        "difficulty":         task["difficulty"],
        "score":              score,
        "episode_cost_usd":   round(summary["episode_cost_usd"], 4),
        "solar_used_kwh":     round(summary["episode_solar_used_kwh"], 4),
        "total_solar_kwh":    round(summary["total_solar_kwh"], 4),
        "curtailed_kwh":      round(summary["episode_curtailed_kwh"], 4),
        "steps_completed":    len(summary["soc_trajectory"]),
        "elapsed_seconds":    round(elapsed, 2),
        "timestamp":          _now(),
    }), flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_task(client, model: str, task: dict, max_steps: int, log_every: int = 8) -> float:
    env = InlineMicrogridEnv()
    obs = env.reset(seed=task["seed"])

    log_start(task)
    t0 = time.time()
    step = 0
    total_r = 0.0

    while not obs.done and step < max_steps:
        action = _llm_act(client, model, obs)
        obs, reward, done, _ = env.step(*action)
        total_r += reward

        if step % log_every == 0 or obs.done:
            log_step(step, obs, action, reward, 0.0)
        step += 1

    summary = env.episode_summary()
    score   = _grade(task["task_id"], summary)
    elapsed = time.time() - t0
    log_end(task, score, summary, elapsed)
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",   default=None, help="Run a single task by task_id")
    parser.add_argument("--steps",  type=int, default=96, help="Max steps per episode (default 96)")
    parser.add_argument("--every",  type=int, default=8,  help="Log every N steps (default 8)")
    args = parser.parse_args()

    # Validate required env vars
    if not HF_TOKEN:
        print("[WARN] HF_TOKEN not set — API calls may fail.", file=sys.stderr)

    try:
        from openai import OpenAI
    except ImportError:
        print("[ERROR] openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=HF_TOKEN or "placeholder", base_url=API_BASE_URL)

    tasks = TASKS
    if args.task:
        tasks = [t for t in TASKS if t["task_id"] == args.task]
        if not tasks:
            print(f"[ERROR] Unknown task '{args.task}'. Valid: {[t['task_id'] for t in TASKS]}", file=sys.stderr)
            sys.exit(1)

    print(json.dumps({
        "event":       "[INIT]",
        "model":       MODEL_NAME,
        "api_base":    API_BASE_URL,
        "tasks":       [t["task_id"] for t in tasks],
        "max_steps":   args.steps,
        "timestamp":   _now(),
    }), flush=True)

    results = {}
    t_total = time.time()

    for task in tasks:
        score = run_task(client, MODEL_NAME, task, max_steps=args.steps, log_every=args.every)
        results[task["task_id"]] = score
        print(json.dumps({
            "event":      "[SCORE]",
            "task_id":    task["task_id"],
            "difficulty": task["difficulty"],
            "score":      score,
        }), flush=True)

    total_elapsed = time.time() - t_total
    print(json.dumps({
        "event":         "[SUMMARY]",
        "scores":        results,
        "mean_score":    round(sum(results.values()) / len(results), 4) if results else 0.0,
        "elapsed_total": round(total_elapsed, 2),
        "timestamp":     _now(),
    }), flush=True)


if __name__ == "__main__":
    main()

"""
Test the PowerGrid environment using HuggingFace Inference API.
No local GPU or model download needed — calls HF's hosted models directly.

Usage:
    python test_with_hf_api.py --token hf_xxxx
    python test_with_hf_api.py --token hf_xxxx --model Qwen/Qwen2.5-3B-Instruct
    python test_with_hf_api.py --token hf_xxxx --episodes 5 --verbose
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass

# Load .env if present (no dependency needed — manual parse)
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ── Inline environment (no server needed) ───────────────────────────────────

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


def _hour(t):    return t * DT
def _solar(t, noise=0.05):
    h = _hour(t)
    if h < 6 or h > 18: return 0.0
    return max(0.0, SOLAR_PEAK_KW * math.exp(-((h-12)**2)/8) * (1 + noise*random.gauss(0,1)))
def _load(t, noise=0.05):
    h = _hour(t)
    return max(0.5, (3+8*math.exp(-((h-8)**2)/2)+12*math.exp(-((h-19)**2)/2))*(1+noise*random.gauss(0,0.5)))
def _price(t):
    h = _hour(t)
    if (7<=h<11) or (17<=h<21): return 0.30
    if 11<=h<17: return 0.15
    return 0.08


@dataclass
class Obs:
    step: int; hour: float; solar_kw: float; demand_kw: float
    battery_soc: float; price: float; grid_import_kw: float
    episode_cost: float; done: bool; reward: float; context: str = ""


class MicrogridEnv:
    def reset(self, seed=None):
        if seed is not None: random.seed(seed)
        self.t = 0; self.soc = random.uniform(0.2, 0.8)
        self.cost = 0.0; self.done = False
        self._s = [_solar(t) for t in range(STEPS)]
        self._l = [_load(t)  for t in range(STEPS)]
        self._p = [_price(t) for t in range(STEPS)]
        return self._obs(0, 0, 0)

    def step(self, batt_cmd, curtail):
        batt_cmd = max(-1.0, min(1.0, float(batt_cmd)))
        curtail  = max(0.0,  min(1.0, float(curtail)))
        t = self.t
        s, l, p = self._s[t], self._l[t], self._p[t]
        flex = l * FLEX_FRAC

        desired = batt_cmd * BATTERY_MAX_POWER_KW
        batt_kw, self.soc = self._battery(desired, self.soc)

        curtailed = curtail * flex
        served    = l - curtailed
        net       = s - served - batt_kw
        imp       = max(0.0, -net)
        exp       = max(0.0,  net)

        r = -(imp*p*DT - exp*p*EXPORT_RATIO*DT)   # cost reward
        r -= curtailed * DT * CURTAIL_PENALTY       # curtailment penalty
        r += (0.05*min(1,(s-exp)/max(s,0.01))*s*DT) if s>0.5 else 0  # solar bonus
        soc_m = min(self.soc-BATTERY_SOC_MIN, BATTERY_SOC_MAX-self.soc)
        r += 0 if soc_m>0.05 else -0.5*(0.05-soc_m)/0.05

        self.cost += imp*p*DT
        self.t += 1
        if self.t >= STEPS:
            self.done = True
            r += max(-2.0, min(5.0, 3.0 - 0.1*self.cost))
        return self._obs(imp, r, batt_kw), r, self.done, {}

    def _battery(self, desired, soc):
        if desired > 0:
            mk = min(BATTERY_MAX_POWER_KW, (BATTERY_SOC_MAX-soc)*BATTERY_CAPACITY_KWH/(DT*BATTERY_EFFICIENCY))
            a  = min(desired, mk)
            s2 = soc + a*DT*BATTERY_EFFICIENCY/BATTERY_CAPACITY_KWH
        elif desired < 0:
            mk = min(BATTERY_MAX_POWER_KW, (soc-BATTERY_SOC_MIN)*BATTERY_CAPACITY_KWH*BATTERY_EFFICIENCY/DT)
            a  = max(desired, -mk)
            s2 = soc - (-a)*DT/BATTERY_EFFICIENCY/BATTERY_CAPACITY_KWH
        else:
            a, s2 = 0.0, soc
        return a, max(BATTERY_SOC_MIN, min(BATTERY_SOC_MAX, s2))

    def _obs(self, imp, reward, batt_kw):
        t  = min(self.t, STEPS-1)
        h  = _hour(t); hi=int(h); mi=int((h-hi)*60)
        p  = self._p[t]
        pl = "ON-PEAK" if p>=0.30 else "MID-PEAK" if p>=0.15 else "OFF-PEAK"
        ctx = (f"Time {hi:02d}:{mi:02d} | Step {self.t}/96 | "
               f"Solar {self._s[t]:.1f} kW | Demand {self._l[t]:.1f} kW | "
               f"Battery SoC {self.soc*100:.0f}% | Price {p:.2f} $/kWh ({pl}) | "
               f"Cost so far ${self.cost:.2f}")
        return Obs(step=self.t, hour=h, solar_kw=self._s[t], demand_kw=self._l[t],
                   battery_soc=self.soc, price=p, grid_import_kw=imp,
                   episode_cost=self.cost, done=self.done, reward=reward, context=ctx)


# ── HuggingFace Inference API agent ─────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert power grid operator managing a residential microgrid.
Minimise electricity costs over a 24-hour period by controlling battery storage.

Grid parameters:
- Battery: 50 kWh capacity, max 25 kW charge/discharge
- Solar PV: up to 20 kW peak at solar noon (10:00-14:00)
- Prices: OFF-PEAK $0.08 (21:00-07:00), MID-PEAK $0.15, ON-PEAK $0.30 (07:00-11:00, 17:00-21:00)

Respond with ONLY a JSON object (no explanation):
{"battery_action": <float -1 to 1>, "curtailment": <float 0 to 1>}
battery_action: -1=full discharge, 0=idle, +1=full charge
curtailment: 0=no shedding (prefer 0 unless desperate)"""


def make_user_prompt(obs: Obs) -> str:
    steps_left = 96 - obs.step
    hint = (
        "DISCHARGE battery — peak pricing now!" if obs.price >= 0.30 and obs.battery_soc > 0.3 else
        "CHARGE from solar surplus — good opportunity" if obs.solar_kw > 8 and obs.price <= 0.15 else
        "CHARGE from grid — cheap off-peak electricity" if obs.price < 0.10 and obs.battery_soc < 0.6 else
        "HOLD — no strong arbitrage signal"
    )
    return f"""Current microgrid state:
{obs.context}
Steps remaining: {steps_left}/96
Hint: {hint}

Your JSON action:"""


def parse_action(text: str):
    """Parse battery_action and curtailment from LLM response. Robust to noise."""
    try:
        m = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if m:
            d = json.loads(m.group())
            return (
                max(-1.0, min(1.0, float(d.get("battery_action", 0.0)))),
                max(0.0,  min(1.0, float(d.get("curtailment",    0.0)))),
            )
    except Exception:
        pass
    # Regex fallback
    bm = re.search(r'battery_action["\s:]+([-\d.]+)', text)
    cm = re.search(r'curtailment["\s:]+([\d.]+)', text)
    batt = max(-1.0, min(1.0, float(bm.group(1)))) if bm else 0.0
    curt = max(0.0,  min(1.0, float(cm.group(1)))) if cm else 0.0
    return batt, curt


class HFAgent:
    def __init__(self, token: str, model: str):
        from huggingface_hub import InferenceClient
        # v1.x uses api_key=; older versions use token=
        try:
            self.client = InferenceClient(api_key=token)
        except TypeError:
            self.client = InferenceClient(token=token)
        self.model  = model
        self.token  = token
        self.calls  = 0
        self.errors = 0

    def act(self, obs: Obs, verbose: bool = False) -> tuple[float, float]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": make_user_prompt(obs)},
        ]
        # Try method 1: OpenAI-compatible API (huggingface_hub >= 0.22)
        try:
            response = self.client.chat.completions.create(
                model       = self.model,
                messages    = messages,
                max_tokens  = 60,
                temperature = 0.1,
            )
            text = response.choices[0].message.content.strip()
            self.calls += 1
            if verbose:
                print(f"  LLM → {text}")
            return parse_action(text)
        except Exception as e1:
            pass

        # Try method 2: legacy chat_completion
        try:
            response = self.client.chat_completion(
                model       = self.model,
                messages    = messages,
                max_tokens  = 60,
                temperature = 0.1,
            )
            text = response.choices[0].message.content.strip()
            self.calls += 1
            if verbose:
                print(f"  LLM → {text}")
            return parse_action(text)
        except Exception as e2:
            pass

        # Try method 3: text_generation with manual prompt formatting
        try:
            prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{make_user_prompt(obs)}\n<|assistant|>\n"
            text = self.client.text_generation(
                prompt          = prompt,
                model           = self.model,
                max_new_tokens  = 60,
                temperature     = 0.1,
            )
            self.calls += 1
            if verbose:
                print(f"  LLM → {text}")
            return parse_action(text)
        except Exception as e3:
            self.errors += 1
            if self.errors <= 3:  # only print first 3 errors to avoid spam
                print(f"  [API ERROR] All methods failed.")
                print(f"    Method 1 (chat.completions): {e1}")
                print(f"    Method 2 (chat_completion):  {e2}")
                print(f"    Method 3 (text_generation):  {e3}")
                if self.errors == 3:
                    print("  (suppressing further errors...)")
            return 0.0, 0.0


# ── Baselines ────────────────────────────────────────────────────────────────

def random_policy(obs):
    return random.uniform(-1, 1), random.uniform(0, 0.2)

def rule_policy(obs):
    h, soc = obs.hour, obs.battery_soc
    if 17 <= h < 21 and soc > 0.3:  return -0.8, 0.0
    if 10 <= h < 15 and obs.solar_kw > 5: return 0.7, 0.0
    if obs.price >= 0.30 and soc > 0.2:   return -0.4, 0.0
    return 0.0, 0.0


def run_episode(env, policy_fn, seed=None, verbose=False):
    obs = env.reset(seed=seed)
    total_r = 0.0
    while not obs.done:
        if verbose:
            print(f"  {obs.context}")
        batt, curt = policy_fn(obs)
        obs, r, done, _ = env.step(batt, curt)
        total_r += r
    return total_r, obs.episode_cost


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test PowerGrid env with HuggingFace Inference API")
    parser.add_argument("--token",    default=os.environ.get("HF_TOKEN"), help="HuggingFace API token (hf_xxx) — or set HF_TOKEN in .env")
    parser.add_argument("--model",    default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HF model ID to use (default: Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run (default: 10)")
    parser.add_argument("--delay",    type=float, default=0.3, help="Seconds between API calls to avoid rate limit (default: 0.3)")
    parser.add_argument("--verbose",  action="store_true", help="Print each step and LLM response")
    args = parser.parse_args()

    env   = MicrogridEnv()
    agent = HFAgent(token=args.token, model=args.model)

    if not args.token:
        print("ERROR: No HF token found. Either:")
        print("  1. Add HF_TOKEN=hf_xxx to your .env file")
        print("  2. Run with: python test_with_hf_api.py --token hf_xxx")
        sys.exit(1)

    print(f"\nModel : {args.model}")
    print(f"Episodes: {args.episodes}\n")
    print("─" * 60)

    # Baselines (fast, no API calls)
    print("Running baselines...")
    rand_rewards = [run_episode(env, random_policy, seed=i)[0] for i in range(10)]
    rule_rewards = [run_episode(env, rule_policy,   seed=i)[0] for i in range(10)]
    import statistics
    print(f"  Random agent    : {statistics.mean(rand_rewards):.2f} avg reward")
    print(f"  Rule-based      : {statistics.mean(rule_rewards):.2f} avg reward")
    print("─" * 60)

    # LLM agent episodes
    llm_rewards, llm_costs = [], []
    for ep in range(args.episodes):
        print(f"\nEpisode {ep+1}/{args.episodes}")
        obs = env.reset(seed=ep + 100)
        total_r = 0.0
        step = 0
        while not obs.done:
            if args.verbose or step % 16 == 0:  # print every 4 hours
                print(f"  {obs.context}")
            batt, curt = agent.act(obs, verbose=args.verbose)
            if not args.verbose and step % 16 == 0:
                print(f"    → battery={batt:+.2f}  curtail={curt:.2f}")
            obs, r, done, _ = env.step(batt, curt)
            total_r += r
            step += 1
            if args.delay > 0:
                time.sleep(args.delay)

        llm_rewards.append(total_r)
        llm_costs.append(obs.episode_cost)
        print(f"  Reward: {total_r:.2f}  |  Energy cost: ${obs.episode_cost:.2f}")

    print("\n" + "═" * 60)
    print("RESULTS SUMMARY")
    print("─" * 60)
    print(f"  Random agent    : {statistics.mean(rand_rewards):.2f} avg reward")
    print(f"  Rule-based      : {statistics.mean(rule_rewards):.2f} avg reward")
    print(f"  HF LLM ({args.model.split('/')[-1][:20]:<20}): {statistics.mean(llm_rewards):.2f} avg reward")
    print(f"  HF LLM avg cost : ${statistics.mean(llm_costs):.2f}/day")
    print(f"  API calls made  : {agent.calls}  |  Errors: {agent.errors}")
    print("═" * 60)


if __name__ == "__main__":
    main()

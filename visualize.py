"""
PowerGrid visualization: baselines always run; LLM agent runs if --token is provided.

Usage:
    python visualize.py                         # baselines only
    python visualize.py --token hf_xxx          # + LLM agent
    python visualize.py --token hf_xxx --episodes 5
"""

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import time

# ── Inline environment ────────────────────────────────────────────────────────

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


def _hour(t):
    return t * DT

def _solar(t, noise=0.05):
    h = _hour(t)
    if h < 6 or h > 18:
        return 0.0
    return max(0.0, SOLAR_PEAK_KW * math.exp(-((h - 12) ** 2) / 8) * (1 + noise * random.gauss(0, 1)))

def _load(t, noise=0.05):
    h = _hour(t)
    return max(0.5, (3 + 8 * math.exp(-((h - 8) ** 2) / 2) + 12 * math.exp(-((h - 19) ** 2) / 2)) * (1 + noise * random.gauss(0, 0.5)))

def _price(t):
    h = _hour(t)
    if (7 <= h < 11) or (17 <= h < 21):
        return 0.30
    if 11 <= h < 17:
        return 0.15
    return 0.08


class MicrogridEnv:
    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.t = 0
        self.soc = random.uniform(0.2, 0.8)
        self.cost = 0.0
        self.done = False
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
        exp_      = max(0.0,  net)

        _msc  = 50.0 * 0.30 * DT                                # max step cost = 3.75
        net_cost = imp*p*DT - exp_*p*EXPORT_RATIO*DT
        r_cost  = max(0.0, min(1.0, 1.0 - net_cost / _msc))
        r_solar = max(0.0, min(1.0, (s - exp_) / s)) if s > 0.5 else 1.0
        r_svc   = min(1.0, served / max(l, 0.01))
        soc_m   = min(self.soc - BATTERY_SOC_MIN, BATTERY_SOC_MAX - self.soc)
        r_stab  = max(0.0, min(1.0, soc_m / 0.10))
        r = 0.60*r_cost + 0.20*r_solar + 0.10*r_svc + 0.10*r_stab

        self.cost += max(0.0, net_cost)
        self.t += 1
        if self.t >= STEPS:
            self.done = True
            sol_kwh = sum(self._s) * DT
            sol_ratio = self.cost / max(sol_kwh, 0.1)  # crude proxy; terminal is minor
            r = min(1.0, r + 0.05 * max(0.0, min(1.0, 1.0 - sol_ratio / 10.0)))
        return self._obs(imp, r, batt_kw), r, self.done, {}

    def _battery(self, desired, soc):
        if desired > 0:
            mk = min(BATTERY_MAX_POWER_KW, (BATTERY_SOC_MAX - soc) * BATTERY_CAPACITY_KWH / (DT * BATTERY_EFFICIENCY))
            a  = min(desired, mk)
            s2 = soc + a * DT * BATTERY_EFFICIENCY / BATTERY_CAPACITY_KWH
        elif desired < 0:
            mk = min(BATTERY_MAX_POWER_KW, (soc - BATTERY_SOC_MIN) * BATTERY_CAPACITY_KWH * BATTERY_EFFICIENCY / DT)
            a  = max(desired, -mk)
            s2 = soc - (-a) * DT / BATTERY_EFFICIENCY / BATTERY_CAPACITY_KWH
        else:
            a, s2 = 0.0, soc
        return a, max(BATTERY_SOC_MIN, min(BATTERY_SOC_MAX, s2))

    def _obs(self, imp, reward, batt_kw):
        t   = min(self.t, STEPS - 1)
        h   = _hour(t)
        hi  = int(h)
        mi  = int((h - hi) * 60)
        p   = self._p[t]
        pl  = "ON-PEAK" if p >= 0.30 else "MID-PEAK" if p >= 0.15 else "OFF-PEAK"
        ctx = (f"Time {hi:02d}:{mi:02d} | Step {self.t}/96 | "
               f"Solar {self._s[t]:.1f} kW | Demand {self._l[t]:.1f} kW | "
               f"Battery SoC {self.soc*100:.0f}% | Price {p:.2f} $/kWh ({pl}) | "
               f"Cost so far ${self.cost:.2f}")
        return type("Obs", (), dict(
            step=self.t, hour=h, solar_kw=self._s[t], demand_kw=self._l[t],
            battery_soc=self.soc, price=p, grid_import_kw=imp,
            episode_cost=self.cost, done=self.done, reward=reward,
            context=ctx,
            _solar=self._s, _load=self._l, _price=self._p,
        ))()


# ── Policies ──────────────────────────────────────────────────────────────────

def random_policy(obs):
    return random.uniform(-1, 1), random.uniform(0, 0.2)

def rule_policy(obs):
    h, soc = obs.hour, obs.battery_soc
    if 17 <= h < 21 and soc > 0.3:
        return -0.8, 0.0
    if 10 <= h < 15 and obs.solar_kw > 5:
        return 0.7, 0.0
    if obs.price >= 0.30 and soc > 0.2:
        return -0.4, 0.0
    return 0.0, 0.0


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


def make_user_prompt(obs):
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


def parse_action(text):
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
    bm = re.search(r'battery_action["\s:]+([-\d.]+)', text)
    cm = re.search(r'curtailment["\s:]+([\d.]+)', text)
    return (
        max(-1.0, min(1.0, float(bm.group(1)))) if bm else 0.0,
        max(0.0,  min(1.0, float(cm.group(1)))) if cm else 0.0,
    )


class HFAgent:
    def __init__(self, token, model):
        from huggingface_hub import InferenceClient
        try:
            self.client = InferenceClient(api_key=token)
        except TypeError:
            self.client = InferenceClient(token=token)
        self.model  = model
        self.calls  = 0
        self.errors = 0

    def act(self, obs):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": make_user_prompt(obs)},
        ]
        for method in ("chat_completions", "chat_completion", "text_generation"):
            try:
                if method == "chat_completions":
                    r = self.client.chat.completions.create(
                        model=self.model, messages=messages, max_tokens=60, temperature=0.1)
                    text = r.choices[0].message.content.strip()
                elif method == "chat_completion":
                    r = self.client.chat_completion(
                        model=self.model, messages=messages, max_tokens=60, temperature=0.1)
                    text = r.choices[0].message.content.strip()
                else:
                    prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{make_user_prompt(obs)}\n<|assistant|>\n"
                    text = self.client.text_generation(
                        prompt=prompt, model=self.model, max_new_tokens=60, temperature=0.1)
                self.calls += 1
                return parse_action(text)
            except Exception:
                pass
        self.errors += 1
        return 0.0, 0.0


# ── Episode runners ───────────────────────────────────────────────────────────

def run_episode(env, policy_fn, seed=None, collect_trace=False, delay=0.0):
    obs = env.reset(seed=seed)
    total_r = 0.0
    trace = {"hours": [], "solar": [], "demand": [], "import_kw": [], "soc": [], "price": [], "batt_action": []}
    while not obs.done:
        batt, curt = policy_fn(obs)
        if collect_trace:
            trace["hours"].append(obs.hour)
            trace["solar"].append(obs.solar_kw)
            trace["demand"].append(obs.demand_kw)
            trace["import_kw"].append(obs.grid_import_kw)
            trace["soc"].append(obs.battery_soc * 100)
            trace["price"].append(obs.price)
            trace["batt_action"].append(batt)
        obs, r, done, _ = env.step(batt, curt)
        total_r += r
        if delay > 0:
            time.sleep(delay)
    return total_r, obs.episode_cost, trace


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_results(rand_rewards, rule_rewards, llm_rewards, llm_costs,
                 rand_trace, rule_trace, llm_trace, model_name, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    has_llm = len(llm_rewards) > 0
    colors   = {"Random": "#e74c3c", "Rule-based": "#3498db", "LLM Agent": "#2ecc71"}

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0f1117")
    plt.suptitle("PowerGrid Microgrid Controller — Evaluation Results",
                 color="white", fontsize=16, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35,
                          left=0.06, right=0.97, top=0.93, bottom=0.06)

    ax_style = dict(facecolor="#1a1d27", grid_color="#2a2d3a")

    def styled_ax(ax, title, xlabel="", ylabel=""):
        ax.set_facecolor(ax_style["facecolor"])
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2d3a")
        ax.set_title(title, color="white", fontsize=10, pad=6)
        if xlabel:
            ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=8)
        if ylabel:
            ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=8)
        ax.grid(True, color="#2a2d3a", linewidth=0.5, alpha=0.7)

    # ── Plot 1: Reward distributions (box + scatter) ──────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    styled_ax(ax1, "Episode Reward Distribution", ylabel="Total Reward")
    groups = [("Random", rand_rewards, colors["Random"]),
              ("Rule-based", rule_rewards, colors["Rule-based"])]
    if has_llm:
        groups.append(("LLM Agent", llm_rewards, colors["LLM Agent"]))

    positions = list(range(len(groups)))
    for pos, (label, rewards, col) in zip(positions, groups):
        bp = ax1.boxplot(rewards, positions=[pos], widths=0.5, patch_artist=True,
                         medianprops=dict(color="white", linewidth=2),
                         whiskerprops=dict(color=col), capprops=dict(color=col),
                         flierprops=dict(marker="o", color=col, markersize=4))
        bp["boxes"][0].set(facecolor=col, alpha=0.6)
        jitter = np.random.uniform(-0.12, 0.12, len(rewards))
        ax1.scatter([pos + j for j in jitter], rewards, color=col, s=20, alpha=0.8, zorder=5)

    ax1.set_xticks(positions)
    ax1.set_xticklabels([g[0] for g in groups], color="white", fontsize=8)

    # ── Plot 2: Mean reward bar chart ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    styled_ax(ax2, "Mean Episode Reward", ylabel="Avg Reward")
    means  = [statistics.mean(r) for _, r, _ in groups]
    stdevs = [statistics.stdev(r) if len(r) > 1 else 0 for _, r, _ in groups]
    bars   = ax2.bar([g[0] for g in groups], means, color=[g[2] for g in groups],
                     alpha=0.8, width=0.5, zorder=3)
    ax2.errorbar([g[0] for g in groups], means, yerr=stdevs,
                 fmt="none", color="white", capsize=4, linewidth=1.5)
    for bar, val in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")
    ax2.tick_params(axis="x", colors="white")

    # ── Plot 3: Cost comparison ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    styled_ax(ax3, "Mean Daily Energy Cost", ylabel="Cost ($/day)")
    rand_costs_ep = [run_episode(MicrogridEnv(), random_policy, seed=i)[1] for i in range(10)]
    rule_costs_ep = [run_episode(MicrogridEnv(), rule_policy,   seed=i)[1] for i in range(10)]
    cost_groups = [("Random", rand_costs_ep, colors["Random"]),
                   ("Rule-based", rule_costs_ep, colors["Rule-based"])]
    if has_llm:
        cost_groups.append(("LLM Agent", llm_costs, colors["LLM Agent"]))
    cost_means = [statistics.mean(c) for _, c, _ in cost_groups]
    cbars = ax3.bar([g[0] for g in cost_groups], cost_means, color=[g[2] for g in cost_groups],
                    alpha=0.8, width=0.5, zorder=3)
    for bar, val in zip(cbars, cost_means):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"${val:.2f}", ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")
    ax3.tick_params(axis="x", colors="white")

    # ── Plots 4-6: 24-hour traces ─────────────────────────────────────────────
    trace_configs = [
        (rand_trace,  "Random Agent",    colors["Random"],    gs[1, 0], gs[2, 0]),
        (rule_trace,  "Rule-based Agent", colors["Rule-based"], gs[1, 1], gs[2, 1]),
    ]
    if has_llm:
        trace_configs.append((llm_trace, f"LLM Agent ({model_name})", colors["LLM Agent"], gs[1, 2], gs[2, 2]))

    for trace, label, col, gs_top, gs_bot in trace_configs:
        if not trace["hours"]:
            continue
        hours = trace["hours"]

        # Power flows
        ax_top = fig.add_subplot(gs_top)
        styled_ax(ax_top, f"{label} — Power Flows", ylabel="Power (kW)")
        ax_top.plot(hours, trace["solar"],     color="#f39c12", linewidth=1.5, label="Solar")
        ax_top.plot(hours, trace["demand"],    color="#9b59b6", linewidth=1.5, label="Demand")
        ax_top.plot(hours, trace["import_kw"], color=col,       linewidth=1.5, label="Grid Import")
        ax_top.fill_between(hours, 0, trace["import_kw"], color=col, alpha=0.15)

        # Price shading
        for i in range(len(hours) - 1):
            p = trace["price"][i]
            shade = "#ff000020" if p >= 0.30 else "#ffff0010" if p >= 0.15 else "#00ff0008"
            ax_top.axvspan(hours[i], hours[i + 1], color=shade, linewidth=0)

        ax_top.legend(loc="upper left", fontsize=7, facecolor="#1a1d27",
                      labelcolor="white", framealpha=0.8)
        ax_top.set_xlim(0, 24)
        ax_top.set_xticks(range(0, 25, 4))
        ax_top.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 4)], fontsize=7)

        # Battery SoC
        ax_bot = fig.add_subplot(gs_bot)
        styled_ax(ax_bot, f"{label} — Battery SoC", xlabel="Hour of Day", ylabel="SoC (%)")
        ax_bot.plot(hours, trace["soc"], color=col, linewidth=2)
        ax_bot.fill_between(hours, BATTERY_SOC_MIN * 100, trace["soc"], color=col, alpha=0.25)
        ax_bot.axhline(BATTERY_SOC_MIN * 100, color="#e74c3c", linestyle="--", linewidth=0.8, alpha=0.7)
        ax_bot.axhline(BATTERY_SOC_MAX * 100, color="#2ecc71", linestyle="--", linewidth=0.8, alpha=0.7)
        ax_bot.set_ylim(0, 100)
        ax_bot.set_xlim(0, 24)
        ax_bot.set_xticks(range(0, 25, 4))
        ax_bot.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 4)], fontsize=7)

    # Fill empty trace slots if LLM not run
    if not has_llm:
        for slot in (gs[1, 2], gs[2, 2]):
            ax_empty = fig.add_subplot(slot)
            ax_empty.set_facecolor("#1a1d27")
            ax_empty.text(0.5, 0.5, "LLM Agent\n(no token provided)",
                          ha="center", va="center", color="#555555",
                          fontsize=10, transform=ax_empty.transAxes)
            for spine in ax_empty.spines.values():
                spine.set_edgecolor("#2a2d3a")
            ax_empty.set_xticks([])
            ax_empty.set_yticks([])

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nVisualization saved: {out_path}")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load .env
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(_env_path):
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip())

    parser = argparse.ArgumentParser(description="PowerGrid visualizer")
    parser.add_argument("--token",    default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace API token (optional — baselines always run)")
    parser.add_argument("--model",    default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--episodes", type=int, default=5,
                        help="LLM episodes (default 5; baselines always use 10)")
    parser.add_argument("--delay",    type=float, default=0.3)
    parser.add_argument("--out",      default="powergrid_results.png")
    args = parser.parse_args()

    env = MicrogridEnv()

    # Baselines
    print("Running Random baseline  (10 episodes)...")
    rand_rewards, rand_costs, rand_trace = [], [], {}
    for i in range(10):
        r, c, tr = run_episode(env, random_policy, seed=i, collect_trace=(i == 0))
        rand_rewards.append(r)
        rand_costs.append(c)
        if i == 0:
            rand_trace = tr
    print(f"  Random     avg reward {statistics.mean(rand_rewards):.2f}  cost ${statistics.mean(rand_costs):.2f}")

    print("Running Rule-based baseline (10 episodes)...")
    rule_rewards, rule_costs, rule_trace = [], [], {}
    for i in range(10):
        r, c, tr = run_episode(env, rule_policy, seed=i, collect_trace=(i == 0))
        rule_rewards.append(r)
        rule_costs.append(c)
        if i == 0:
            rule_trace = tr
    print(f"  Rule-based avg reward {statistics.mean(rule_rewards):.2f}  cost ${statistics.mean(rule_costs):.2f}")

    # LLM agent (optional)
    llm_rewards, llm_costs, llm_trace = [], [], {}
    has_real_token = args.token and args.token not in ("hf_xxx", "hf_your_token_here", "")
    if has_real_token:
        print(f"\nRunning LLM agent ({args.model}, {args.episodes} episodes)...")
        agent = HFAgent(token=args.token, model=args.model)
        for ep in range(args.episodes):
            collect = (ep == 0)
            r, c, tr = run_episode(env,
                                   lambda obs, a=agent: a.act(obs),
                                   seed=ep + 100,
                                   collect_trace=collect,
                                   delay=args.delay)
            llm_rewards.append(r)
            llm_costs.append(c)
            if collect:
                llm_trace = tr
            print(f"  Episode {ep+1}: reward={r:.2f}  cost=${c:.2f}")
        print(f"  LLM avg reward {statistics.mean(llm_rewards):.2f}  cost ${statistics.mean(llm_costs):.2f}")
        print(f"  API calls: {agent.calls}  errors: {agent.errors}")
    else:
        print("\nNo valid HF token - skipping LLM agent.")
        print("  Run with: python visualize.py --token hf_xxx")

    # Results table
    print("\n" + "=" * 55)
    print(f"{'Agent':<25} {'Avg Reward':>12} {'Avg Cost':>12}")
    print("-" * 55)
    print(f"{'Random':<25} {statistics.mean(rand_rewards):>12.2f} ${statistics.mean(rand_costs):>10.2f}")
    print(f"{'Rule-based':<25} {statistics.mean(rule_rewards):>12.2f} ${statistics.mean(rule_costs):>10.2f}")
    if llm_rewards:
        print(f"{'LLM Agent':<25} {statistics.mean(llm_rewards):>12.2f} ${statistics.mean(llm_costs):>10.2f}")
    print("=" * 55)

    # Plot
    model_short = args.model.split("/")[-1]
    plot_results(rand_rewards, rule_rewards, llm_rewards, llm_costs,
                 rand_trace, rule_trace, llm_trace, model_short, args.out)


if __name__ == "__main__":
    main()

"""
Run baseline agents and RL agents on the PowerGrid environment.

Usage:
    python run_baselines.py
    python run_baselines.py --episodes 30 --train 200
    python run_baselines.py --synthetic
    python run_baselines.py --verbose
"""

import argparse
import random
import statistics
import os

import numpy as np

from baseline_agents import RandomAgent, ConservativeAgent, QLearningAgent, SARSAAgent, ALL_AGENTS
from data_loader import get_day_profiles, get_dataset, dataset_stats

# ── Inline environment ────────────────────────────────────────────────────────

BATTERY_CAPACITY_KWH = 50.0
BATTERY_MAX_POWER_KW = 25.0
BATTERY_EFFICIENCY   = 0.95
BATTERY_SOC_MIN      = 0.10
BATTERY_SOC_MAX      = 0.95
FLEX_FRAC            = 0.30
EXPORT_RATIO         = 0.40
STEPS                = 96
DT                   = 0.25
_MSC                 = BATTERY_CAPACITY_KWH * 0.30 * DT   # max step cost anchor


class GridEnv:
    def reset(self, solar, load, price, seed=None):
        if seed is not None:
            random.seed(seed)
        self.t    = 0
        self.soc  = random.uniform(0.2, 0.8)
        self.cost = 0.0
        self.done = False
        self._s, self._l, self._p = solar, load, price
        return self._obs(0.0, 0.0)

    def step(self, batt_cmd, curtail):
        batt_cmd = max(-1.0, min(1.0, float(batt_cmd)))
        curtail  = max(0.0,  min(1.0, float(curtail)))
        t        = self.t
        s, l, p  = self._s[t], self._l[t], self._p[t]

        batt_kw, self.soc = self._battery(batt_cmd * BATTERY_MAX_POWER_KW, self.soc)
        curtailed = curtail * l * FLEX_FRAC
        served    = l - curtailed
        net       = s - served - batt_kw
        imp       = max(0.0, -net)
        exp_      = max(0.0,  net)

        net_cost = imp * p * DT - exp_ * p * EXPORT_RATIO * DT
        r_cost   = max(0.0, min(1.0, 1.0 - net_cost / _MSC))
        r_solar  = max(0.0, min(1.0, (s - exp_) / s)) if s > 0.5 else 1.0
        r_svc    = min(1.0, served / max(l, 0.01))
        soc_m    = min(self.soc - BATTERY_SOC_MIN, BATTERY_SOC_MAX - self.soc)
        r_stab   = max(0.0, min(1.0, soc_m / 0.10))
        reward   = 0.60 * r_cost + 0.20 * r_solar + 0.10 * r_svc + 0.10 * r_stab

        self.cost += max(0.0, net_cost)
        self.t += 1
        if self.t >= STEPS:
            self.done = True
            reward = min(1.0, reward + 0.05)

        return self._obs(imp, reward), reward, self.done, {"import_kw": imp, "batt_kw": batt_kw}

    def _battery(self, desired, soc):
        if desired > 0:
            mk = min(BATTERY_MAX_POWER_KW,
                     (BATTERY_SOC_MAX - soc) * BATTERY_CAPACITY_KWH / (DT * BATTERY_EFFICIENCY))
            a  = min(desired, mk)
            s2 = soc + a * DT * BATTERY_EFFICIENCY / BATTERY_CAPACITY_KWH
        elif desired < 0:
            mk = min(BATTERY_MAX_POWER_KW,
                     (soc - BATTERY_SOC_MIN) * BATTERY_CAPACITY_KWH * BATTERY_EFFICIENCY / DT)
            a  = max(desired, -mk)
            s2 = soc - (-a) * DT / BATTERY_EFFICIENCY / BATTERY_CAPACITY_KWH
        else:
            a, s2 = 0.0, soc
        return a, max(BATTERY_SOC_MIN, min(BATTERY_SOC_MAX, s2))

    def _obs(self, imp, reward):
        t  = min(self.t, STEPS - 1)
        h  = t * DT
        hi = int(h)
        p  = self._p[t]
        pl = "ON-PEAK" if p >= 0.30 else "MID-PEAK" if p >= 0.15 else "OFF-PEAK"
        return type("Obs", (), dict(
            step=self.t, hour=h, solar_kw=self._s[t], demand_kw=self._l[t],
            battery_soc=self.soc, price=p, grid_import_kw=imp,
            episode_cost=self.cost, done=self.done, reward=reward,
            context=(f"Time {hi:02d}:00 | Step {self.t}/96 | "
                     f"Solar {self._s[t]:.1f} kW | Demand {self._l[t]:.1f} kW | "
                     f"SoC {self.soc*100:.0f}% | {p:.2f} $/kWh ({pl}) | "
                     f"Cost ${self.cost:.2f}"),
        ))()


# ── Episode runners ───────────────────────────────────────────────────────────

def train_episode(env, agent, solar, load, price, seed):
    """Run one training episode, calling agent.update(). Returns avg step score."""
    obs          = env.reset(solar, load, price, seed=seed)
    step_rewards = []
    while not obs.done:
        batt, curt            = agent.act(obs)
        next_obs, r, done, _  = env.step(batt, curt)
        agent.update(r, next_obs, done)
        step_rewards.append(r)
        obs = next_obs
    return statistics.mean(step_rewards) if step_rewards else 0.0


def run_episode(env, agent, solar, load, price, seed, verbose=False):
    """Run one evaluation episode, no weight updates."""
    obs          = env.reset(solar, load, price, seed=seed)
    total_r      = 0.0
    step_rewards = []
    step         = 0

    while not obs.done:
        batt, curt       = agent.act(obs)
        obs, r, done, _  = env.step(batt, curt)
        total_r         += r
        step_rewards.append(r)
        if verbose:
            print(f"    step {step:02d}: {obs.context}  | batt={batt:+.2f} r={r:.4f}")
        step += 1

    return {
        "reward":       total_r,
        "score":        statistics.mean(step_rewards),   # avg step reward in [0, 1]
        "cost":         obs.episode_cost,
        "step_rewards": step_rewards,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def _sep(c="-", w=80):
    print(c * w)


def _print_episode_header():
    print(f"  {'Ep':>3}  {'Total R':>8}  {'Score [0-1]':>12}  {'Cost $':>8}")
    _sep()


def _print_episode_row(ep, res):
    print(f"  {ep:>3}  {res['reward']:>8.2f}  {res['score']:>12.4f}  {res['cost']:>8.2f}")


def _print_summary(results_by_agent):
    _sep("=", 80)
    print(f"  {'Agent':<16} {'Eps':>4} {'Avg Score':>10} {'Std':>7} {'Avg Cost':>10} {'Best':>7} {'Worst':>7}")
    _sep("-", 80)
    for name, results in results_by_agent.items():
        scores = [r["score"] for r in results]
        costs  = [r["cost"]  for r in results]
        std    = statistics.stdev(scores) if len(scores) > 1 else 0.0
        print(
            f"  {name:<16} {len(scores):>4} "
            f"{statistics.mean(scores):>10.4f} {std:>7.4f} "
            f"{statistics.mean(costs):>10.2f} "
            f"{max(scores):>7.4f} {min(scores):>7.4f}"
        )
    _sep("=", 80)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",  type=int,   default=30)
    parser.add_argument("--train",     type=int,   default=200,  help="Training episodes for RL agents")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--noise",     type=float, default=0.05)
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--out",       default="outputs/baseline_results.png")
    args = parser.parse_args()

    use_real = not args.synthetic
    np.random.seed(args.seed)
    random.seed(args.seed)

    print()
    _sep("=", 80)
    print("  PowerGrid Baseline & RL Agent Evaluation")
    _sep("=", 80)
    print(f"  Eval episodes   : {args.episodes}")
    print(f"  RL train eps    : {args.train}")
    print(f"  Data source     : {'Real (NREL) + clear-sky solar' if use_real else 'Synthetic'}")
    print(f"  Noise           : {args.noise}  |  Seed: {args.seed}")
    print()

    print("Generating evaluation dataset...")
    eval_data = get_dataset(n_days=args.episodes, noise=args.noise,
                            use_real_data=use_real, seed=args.seed)
    st = dataset_stats(eval_data)
    print(f"  {st['episodes']} episodes | load {st['load_min']:.1f}-{st['load_max']:.1f} kW "
          f"| solar 0-{st['solar_max']:.1f} kW\n")

    print("Generating training dataset for RL agents...")
    train_data = get_dataset(n_days=args.train, noise=args.noise,
                             use_real_data=use_real, seed=args.seed + 9999)
    print(f"  {len(train_data)} episodes\n")

    env    = GridEnv()
    agents = [cls() for cls in ALL_AGENTS]
    results_by_agent = {}

    training_curves = {}   # agent_name -> list of per-episode training scores

    for agent in agents:
        _sep("=", 80)
        print(f"  Agent: {agent.name}")

        # Train RL agents
        if hasattr(agent, "training") and agent.training:
            print(f"  Training for {args.train} episodes...")
            ep_train_scores = []
            for ep, (solar, load, price) in enumerate(train_data):
                score = train_episode(env, agent, solar, load, price,
                                      seed=args.seed + 9999 + ep)
                ep_train_scores.append(score)
            training_curves[agent.name] = ep_train_scores
            agent.set_eval()
            print(f"  Training complete. Evaluating...\n")
        else:
            print()

        _sep("-", 80)
        _print_episode_header()

        ep_results = []
        for ep, (solar, load, price) in enumerate(eval_data):
            res = run_episode(env, agent, solar, load, price,
                              seed=args.seed + ep, verbose=args.verbose)
            ep_results.append(res)
            _print_episode_row(ep + 1, res)

        results_by_agent[agent.name] = ep_results
        scores = [r["score"] for r in ep_results]
        _sep("-", 80)
        print(f"  Avg Score [0-1]: {statistics.mean(scores):.4f}  "
              f"Std: {statistics.stdev(scores) if len(scores) > 1 else 0:.4f}  "
              f"Avg Cost: ${statistics.mean(r['cost'] for r in ep_results):.2f}")
        print()

    print("\nFINAL SUMMARY")
    _print_summary(results_by_agent)

    scores_by_agent = {n: statistics.mean(r["score"] for r in res)
                       for n, res in results_by_agent.items()}
    best = max(scores_by_agent, key=scores_by_agent.get)
    print(f"\n  Best agent : {best}  (avg score {scores_by_agent[best]:.4f})")
    rl_scores = {n: s for n, s in scores_by_agent.items()
                 if n in ("Q-Learning", "SARSA")}
    if rl_scores:
        rl_best = max(rl_scores.values())
        print(f"  RL target  : LLM agent must beat {rl_best:.4f} to show learning")

    try:
        _plot(results_by_agent, eval_data, args.out)
        _plot_training_curves(training_curves, args.out)
    except Exception as e:
        print(f"\n[Warning] Plot failed: {e}")


# ── Training curve plot ───────────────────────────────────────────────────────

def _plot_training_curves(training_curves: dict, eval_out_path: str):
    """Save RL training reward curves — real evidence of learning."""
    if not training_curves:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir  = os.path.dirname(eval_out_path) if os.path.dirname(eval_out_path) else "."
    out_path = os.path.join(out_dir, "training_curves.png")
    os.makedirs(out_dir, exist_ok=True)

    COLORS = {"Q-Learning": "#3498db", "SARSA": "#2ecc71"}
    WINDOW = 10

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("RL Agent Training Curves — Score [0-1] per Training Episode",
                 color="white", fontsize=13, fontweight="bold")

    def ax_style(ax, title, xlabel="", ylabel=""):
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white", labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#2a2d3a")
        ax.set_title(title, color="white", fontsize=11, pad=6)
        if xlabel: ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=9)
        if ylabel: ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=9)
        ax.grid(True, color="#2a2d3a", linewidth=0.5, alpha=0.7)

    # -- Plot 1: raw + smoothed training scores --------------------------------
    ax = axes[0]
    ax_style(ax, "Training Score per Episode (raw + smoothed)",
             "Training episode", "Avg step score [0-1]")
    for name, scores in training_curves.items():
        col = COLORS.get(name, "#ffffff")
        eps = range(1, len(scores) + 1)
        ax.plot(eps, scores, color=col, alpha=0.25, linewidth=0.8)
        if len(scores) >= WINDOW:
            ma = np.convolve(scores, np.ones(WINDOW)/WINDOW, mode='valid')
            ax.plot(range(WINDOW, len(scores)+1), ma, color=col,
                    linewidth=2.2, label=f"{name} ({WINDOW}-ep MA)")
        else:
            ax.plot(eps, scores, color=col, linewidth=2.2, label=name)
    ax.set_ylim(0, 1)
    ax.legend(facecolor="#1a1d27", labelcolor="white", fontsize=9)

    # -- Plot 2: cumulative average (learning progress) -----------------------
    ax = axes[1]
    ax_style(ax, "Cumulative Avg Score (learning progress)",
             "Training episode", "Cumulative avg score [0-1]")
    for name, scores in training_curves.items():
        col    = COLORS.get(name, "#ffffff")
        cumavg = [statistics.mean(scores[:i+1]) for i in range(len(scores))]
        ax.plot(range(1, len(cumavg)+1), cumavg, color=col,
                linewidth=2.5, label=name)
    ax.set_ylim(0, 1)
    ax.legend(facecolor="#1a1d27", labelcolor="white", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Training curves saved: {out_path}")


# ── Plot ──────────────────────────────────────────────────────────────────────

def _plot(results_by_agent, dataset, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    COLORS = {
        "Random":       "#e74c3c",
        "Conservative": "#95a5a6",
        "Q-Learning":   "#3498db",
        "SARSA":        "#2ecc71",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("PowerGrid Baseline Evaluation — Score [0-1]",
                 color="white", fontsize=15, fontweight="bold")

    def ax_style(ax, title, xlabel="", ylabel=""):
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2d3a")
        ax.set_title(title, color="white", fontsize=10, pad=5)
        if xlabel: ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=8)
        if ylabel: ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=8)
        ax.grid(True, color="#2a2d3a", linewidth=0.5, alpha=0.7)

    names = list(results_by_agent.keys())

    # Plot 1: Score [0-1] over episodes
    ax = axes[0, 0]
    ax_style(ax, "Score [0-1] Over Episodes", "Episode", "Avg Step Score [0-1]")
    for name, results in results_by_agent.items():
        scores = [r["score"] for r in results]
        col    = COLORS.get(name, "#ffffff")
        ax.plot(range(1, len(scores) + 1), scores, color=col, alpha=0.5, linewidth=1.0, label=name)
        window = min(5, len(scores))
        ma = [statistics.mean(scores[max(0, i-window):i+1]) for i in range(len(scores))]
        ax.plot(range(1, len(ma) + 1), ma, color=col, linewidth=2.2, linestyle="--")
    ax.set_ylim(0, 1)
    ax.legend(facecolor="#1a1d27", labelcolor="white", fontsize=8)

    # Plot 2: Score distribution boxplot
    ax = axes[0, 1]
    ax_style(ax, "Score Distribution [0-1]", ylabel="Score")
    scores_list = [[r["score"] for r in results_by_agent[n]] for n in names]
    for i, (name, sc) in enumerate(zip(names, scores_list)):
        col = COLORS.get(name, "#ffffff")
        bp  = ax.boxplot(sc, positions=[i], widths=0.5, patch_artist=True,
                         medianprops=dict(color="white", linewidth=2),
                         whiskerprops=dict(color=col), capprops=dict(color=col),
                         flierprops=dict(marker="o", color=col, markersize=3))
        bp["boxes"][0].set(facecolor=col, alpha=0.55)
        jitter = [random.uniform(-0.1, 0.1) for _ in sc]
        ax.scatter([i + j for j in jitter], sc, color=col, s=18, alpha=0.8, zorder=5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, color="white", fontsize=8)
    ax.set_ylim(0, 1)

    # Plot 3: Mean cost bar
    ax = axes[0, 2]
    ax_style(ax, "Mean Daily Energy Cost", ylabel="Cost ($/day)")
    means  = [statistics.mean(r["cost"] for r in results_by_agent[n]) for n in names]
    stdevs = [statistics.stdev([r["cost"] for r in results_by_agent[n]])
              if len(results_by_agent[n]) > 1 else 0 for n in names]
    bars = ax.bar(names, means, color=[COLORS.get(n, "#fff") for n in names], alpha=0.8, width=0.5, zorder=3)
    ax.errorbar(names, means, yerr=stdevs, fmt="none", color="white", capsize=4, linewidth=1.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"${val:.2f}", ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", colors="white", labelsize=8)

    # Plot 4: Sample day profile
    ax = axes[1, 0]
    ax_style(ax, "Sample Day Profile (Real Data)", "Hour", "kW")
    solar0, load0, price0 = dataset[0]
    hours = [i * DT for i in range(STEPS)]
    ax.plot(hours, solar0, color="#f39c12", linewidth=1.5, label="Solar PV")
    ax.plot(hours, load0,  color="#9b59b6", linewidth=1.5, label="Demand")
    for i in range(len(hours) - 1):
        shade = "#ff000020" if price0[i] >= 0.30 else "#ffff0010" if price0[i] >= 0.15 else "#00ff0008"
        ax.axvspan(hours[i], hours[i + 1], color=shade, linewidth=0)
    ax.legend(facecolor="#1a1d27", labelcolor="white", fontsize=8)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 4)], fontsize=7)

    # Plot 5: Step reward trace (episode 1) — all agents
    ax = axes[1, 1]
    ax_style(ax, "Step Reward Trace — Episode 1", "Step", "Step Reward [0-1]")
    for name, results in results_by_agent.items():
        sr  = results[0]["step_rewards"]
        col = COLORS.get(name, "#fff")
        ax.plot(range(len(sr)), sr, color=col, linewidth=1.2, alpha=0.85, label=name)
    ax.set_ylim(0, 1)
    ax.legend(facecolor="#1a1d27", labelcolor="white", fontsize=8)

    # Plot 6: Cumulative avg score
    ax = axes[1, 2]
    ax_style(ax, "Cumulative Avg Score (running)", "Episode", "Score [0-1]")
    for name, results in results_by_agent.items():
        scores = [r["score"] for r in results]
        cumavg = [statistics.mean(scores[:i + 1]) for i in range(len(scores))]
        col    = COLORS.get(name, "#fff")
        ax.plot(range(1, len(cumavg) + 1), cumavg, color=col, linewidth=2, label=name)
    ax.set_ylim(0, 1)
    ax.legend(facecolor="#1a1d27", labelcolor="white", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    main()

"""
run_pipeline.py
---------------
Full end-to-end pipeline:
  1. Create ride-hailing Gym environment (10K daily requests)
  2. Train PPO agent (2M steps, β=0.01 entropy)
  3. Evaluate against baselines (fixed-price, rule-based surge)
  4. Ablation study (reward components + entropy coefficient)
  5. Visualisation (learning curve, revenue comparison, price heatmap)

Usage
-----
  python run_pipeline.py --total_timesteps 2000000 --out_dir outputs
  python run_pipeline.py --quick_test    # 100K steps, fast
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Environment ───────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  STEP 1/4 — Ride-Hailing Gym Environment")
    print("═"*60)
    from envs.ride_hailing_env import RideHailingEnv

    env = RideHailingEnv(
        n_daily_requests=10000,
        reward_weights={"revenue": 1.0, "utilisation": 0.5, "rejection": 0.8},
        discrete_actions=True,
        seed=args.seed,
    )
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space     : {env.action_space.n} discrete price levels")
    print(f"  Daily requests   : 10,000 (stochastic Poisson)")

    # Quick sanity check: one episode with random policy
    result = env.reset()
    obs    = result[0] if isinstance(result, tuple) else result
    ep_rev = 0.0
    for _ in range(96):
        action = env.action_space.sample()
        step   = env.step(action)
        if len(step) == 5:
            obs, rew, term, trunc, info = step
            done = term or trunc
        else:
            obs, rew, done, info = step
        ep_rev += info["revenue"]
        if done: break
    print(f"  Random policy baseline revenue: ${ep_rev:.0f}/day")

    # ── Step 2: Train PPO ─────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  STEP 2/4 — PPO Training (entropy β=0.01)")
    print("═"*60)
    from agents.train_ppo import train_ppo

    ppo_model = train_ppo(
        env=RideHailingEnv(seed=args.seed),
        out_dir=out_dir,
        total_timesteps=args.total_timesteps,
        ent_coef=args.ent_coef,
        seed=args.seed,
        verbose=1,
    )

    # ── Step 3: Baseline comparison ───────────────────────────────────────
    print("\n" + "═"*60)
    print("  STEP 3/4 — Baseline Comparison (100 episodes)")
    print("═"*60)
    from baselines.baselines import run_all_baselines

    env_fn = lambda: RideHailingEnv(seed=args.seed + 200)
    comparison = run_all_baselines(
        ppo_model, env_fn, n_episodes=args.n_eval_episodes, seed=args.seed)

    with open(out_dir / "baseline_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # ── Step 4: Ablation study ────────────────────────────────────────────
    if args.run_ablation:
        print("\n" + "═"*60)
        print("  STEP 4/4 — Ablation Study")
        print("═"*60)
        from agents.train_ppo import run_ablation_study

        ablation = run_ablation_study(
            out_dir=out_dir / "ablation",
            timesteps_per_config=max(100_000, args.total_timesteps // 5),
            n_eval_episodes=args.n_eval_episodes,
            seed=args.seed,
        )
    else:
        ablation = {"note": "Run with --run_ablation to compute ablation study"}

    # ── Visualisation ─────────────────────────────────────────────────────
    _plot_results(comparison, out_dir)

    # ── Final summary ──────────────────────────────────────────────────────
    comp = comparison.get("comparison", {})
    ppo  = comparison.get("ppo", {})

    print("\n" + "═"*60)
    print("  PIPELINE COMPLETE")
    print("═"*60)
    print(f"  PPO vs fixed-price  : +{comp.get('ppo_vs_fixed_pct','N/A')}%  (target: +31%)")
    print(f"  PPO vs rule-based   : +{comp.get('ppo_vs_rule_pct','N/A')}%   (target: +18%)")
    print(f"  Mean utilisation    : {ppo.get('mean_utilisation',0):.3f}  (target: >0.78)")
    print(f"  Mean rejection rate : {ppo.get('mean_rej_rate',0):.3f}  (target: <0.12)")
    if args.run_ablation:
        ab = ablation.get("ablation_summary", {})
        print(f"  Util term stability : +{ab.get('stability_improvement_pct','N/A')}%  (target: +22%)")
        entr = ablation.get("entropy_ablation", {})
        print(f"  Entropy sample eff  : +{entr.get('sample_efficiency_improvement_pct','N/A')}%  (target: +35%)")
    print(f"\n  Outputs → {out_dir}")

    summary = {"comparison": comparison, "ablation": ablation}
    with open(out_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def _plot_results(comparison: dict, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir.mkdir(parents=True, exist_ok=True)

        methods  = ["ppo", "fixed_price", "rule_based", "random"]
        labels   = ["PPO (ours)", "Fixed Price", "Rule-Based Surge", "Random"]
        colors   = ["#E8534A", "#3A7DC9", "#27AE60", "#AAA"]
        revenues = [comparison.get(m, {}).get("mean_revenue", 0) for m in methods]
        utils    = [comparison.get(m, {}).get("mean_utilisation", 0) for m in methods]
        rejs     = [comparison.get(m, {}).get("mean_rej_rate", 0) for m in methods]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, vals, title, ylabel in [
            (axes[0], revenues, "Mean Daily Revenue", "Revenue (USD)"),
            (axes[1], [u*100 for u in utils], "Driver Utilisation", "Utilisation (%)"),
            (axes[2], [r*100 for r in rejs],  "Rider Rejection Rate", "Rejection Rate (%)"),
        ]:
            bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.6)
            ax.set_title(title, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.tick_params(axis="x", rotation=20)
            ax.spines[["top","right"]].set_visible(False)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                         f"{val:.0f}" if ylabel.startswith("Rev") else f"{val:.1f}%",
                         ha="center", fontsize=9)

        # Add target lines
        axes[1].axhline(78, color="red", ls="--", lw=1.2, label="Target 78%")
        axes[1].legend(fontsize=9)
        axes[2].axhline(12, color="red", ls="--", lw=1.2, label="Target 12%")
        axes[2].legend(fontsize=9)

        plt.suptitle("PPO Dynamic Pricing vs Baselines — Ride-Hailing Simulation",
                      fontsize=13)
        plt.tight_layout()
        path = out_dir / "comparison_plot.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot → {path}")
    except Exception as e:
        print(f"  (Plot skipped: {e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Dynamic Pricing Agent")
    parser.add_argument("--total_timesteps",   type=int, default=2_000_000)
    parser.add_argument("--out_dir",           default="outputs")
    parser.add_argument("--n_eval_episodes",   type=int, default=100)
    parser.add_argument("--ent_coef",          type=float, default=0.01)
    parser.add_argument("--run_ablation",      action="store_true")
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--quick_test",        action="store_true",
                        help="100K steps, 10 eval episodes")
    args = parser.parse_args()

    if args.quick_test:
        args.total_timesteps = 100_000
        args.n_eval_episodes = 10

    main(args)

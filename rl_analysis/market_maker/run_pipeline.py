#!/usr/bin/env python3
"""
Complete RL Market Making Pipeline
===================================

Pipeline:
  Step 1: Preprocess MBO .dbn.zst -> 1-second snapshots (CSV)
  Step 2: Run Avellaneda-Stoikov baseline (grid search + best trajectory)
  Step 3: Train DQN agent
  Step 4: Train PPO agent
  Step 5: Compare all agents and save results + plots
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

from .data_preprocessor import preprocess_all, load_snapshots
from .env import MarketMakingEnv
from .baseline import AvellanedaStoikov, run_baseline, grid_search_baseline

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor


DATA_DIR = "XNAS-20260227-QVD7UYV7GQ"
SNAPSHOT_DIR = "snapshots"
RESULTS_DIR = "rl_results"
MODEL_DIR = "models"

EPISODE_LENGTH = 3600   # 1 hour
DQN_TIMESTEPS = 100_000
PPO_TIMESTEPS = 200_000


def step1_preprocess():
    print("=" * 60)
    print("STEP 1: Preprocessing MBO data → 1-second snapshots")
    print("=" * 60)
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(
            f"'{DATA_DIR}' not found. Place the MBO data directory here."
        )
    preprocess_all(DATA_DIR, SNAPSHOT_DIR, freq="1s")


def _run_agent_episode(env, model_or_as, test_data, seed=42, is_as=False, as_model=None):
    """Run one episode and collect PnL/inventory/action traces."""
    env_local = MarketMakingEnv(data=test_data, episode_length=EPISODE_LENGTH)
    obs, _ = env_local.reset(seed=seed)
    done = False
    pnl_trace, inv_trace, action_trace = [], [], []

    while not done:
        if is_as:
            mid = env_local.data.iloc[env_local._step_idx]["mid_price"]
            sigma = env_local.data.iloc[env_local._step_idx]["volatility"]
            sigma = sigma if not np.isnan(sigma) and sigma > 0 else 0.001
            t_rem = 1.0 - obs[-1]
            action = as_model.get_action_for_env(
                mid, env_local.inventory, sigma, t_rem,
                env_local.BID_OFFSETS, env_local.ASK_OFFSETS
            )
        else:
            action, _ = model_or_as.predict(obs, deterministic=True)
            action = int(action)

        obs, reward, terminated, truncated, info = env_local.step(action)
        pnl_trace.append(info["total_pnl"])
        inv_trace.append(info["inventory"])
        action_trace.append(action)
        done = terminated or truncated

    stats = env_local.get_episode_stats()
    return stats, pnl_trace, inv_trace, action_trace


def step2_baseline(test_data):
    print("\n" + "=" * 60)
    print("STEP 2: Avellaneda-Stoikov Baseline")
    print("=" * 60)

    env = MarketMakingEnv(data=test_data, episode_length=EPISODE_LENGTH)

    print("Grid searching gamma and k parameters...")
    results = grid_search_baseline(
        env,
        gammas=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
        ks=[0.5, 1.0, 1.5, 2.0, 5.0, 10.0],
    )

    best = results.sort_values("total_pnl", ascending=False).iloc[0]
    print(f"\nBest AS: gamma={best['gamma']}, k={best['k']}, "
          f"PnL={best['total_pnl']:.2f}, Sharpe={best['sharpe']:.2f}")

    as_model = AvellanedaStoikov(gamma=best["gamma"], k=best["k"])
    stats, pnl, inv, actions = _run_agent_episode(
        None, None, test_data, seed=42, is_as=True, as_model=as_model
    )
    stats["gamma"] = float(best["gamma"])
    stats["k"] = float(best["k"])

    return stats, pnl, inv, actions, results


def step3_train_dqn(train_data, test_data):
    print("\n" + "=" * 60)
    print("STEP 3: Training DQN Agent")
    print("=" * 60)
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_env = Monitor(MarketMakingEnv(data=train_data, episode_length=EPISODE_LENGTH))

    model = DQN(
        "MlpPolicy", train_env,
        learning_rate=1e-4, buffer_size=50_000, learning_starts=1_000,
        batch_size=64, gamma=0.99,
        exploration_fraction=0.3, exploration_final_eps=0.05,
        target_update_interval=500,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=1,
    )

    model.learn(total_timesteps=DQN_TIMESTEPS)
    model.save(os.path.join(MODEL_DIR, "dqn_market_maker"))

    stats, pnl, inv, actions = _run_agent_episode(
        None, model, test_data, seed=42
    )
    print(f"DQN: PnL={stats['total_pnl']:.2f}, Sharpe={stats['sharpe']:.2f}, "
          f"Trades={stats['n_trades']}")

    return model, stats, pnl, inv, actions


def step4_train_ppo(train_data, test_data):
    print("\n" + "=" * 60)
    print("STEP 4: Training PPO Agent")
    print("=" * 60)

    train_env = Monitor(MarketMakingEnv(data=train_data, episode_length=EPISODE_LENGTH))

    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=1,
    )

    model.learn(total_timesteps=PPO_TIMESTEPS)
    model.save(os.path.join(MODEL_DIR, "ppo_market_maker"))

    stats, pnl, inv, actions = _run_agent_episode(
        None, model, test_data, seed=42
    )
    print(f"PPO: PnL={stats['total_pnl']:.2f}, Sharpe={stats['sharpe']:.2f}, "
          f"Trades={stats['n_trades']}")

    return model, stats, pnl, inv, actions


def step5_compare(as_stats, as_pnl, as_inv, as_actions,
                  dqn_stats, dqn_pnl, dqn_inv, dqn_actions,
                  ppo_stats, ppo_pnl, ppo_inv, ppo_actions):
    print("\n" + "=" * 60)
    print("STEP 5: Comparison & Visualization")
    print("=" * 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    comparison = pd.DataFrame([
        {"Agent": "Avellaneda-Stoikov", **{k: v for k, v in as_stats.items() if k not in ("gamma", "k")}},
        {"Agent": "DQN", **dqn_stats},
        {"Agent": "PPO", **ppo_stats},
    ])
    cols = ["Agent", "total_pnl", "sharpe", "max_drawdown", "n_trades", "final_inventory", "max_inventory"]
    print("\n" + comparison[cols].to_string(index=False))
    comparison.to_csv(os.path.join(RESULTS_DIR, "comparison.csv"), index=False)

    # --- Plot 1: PnL and Inventory ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(as_pnl, label="Avellaneda-Stoikov", alpha=0.8, lw=1.2)
    axes[0].plot(dqn_pnl, label="DQN", alpha=0.8, lw=1.2)
    axes[0].plot(ppo_pnl, label="PPO", alpha=0.8, lw=1.2)
    axes[0].set_ylabel("Cumulative PnL ($)")
    axes[0].set_title("Market Making Agent Comparison — Test Episode")
    axes[0].legend()
    axes[0].axhline(0, color="gray", ls="--", lw=0.5)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(as_inv, label="Avellaneda-Stoikov", alpha=0.8, lw=1.2)
    axes[1].plot(dqn_inv, label="DQN", alpha=0.8, lw=1.2)
    axes[1].plot(ppo_inv, label="PPO", alpha=0.8, lw=1.2)
    axes[1].set_ylabel("Inventory (shares)")
    axes[1].set_xlabel("Time Step (seconds)")
    axes[1].legend()
    axes[1].axhline(0, color="gray", ls="--", lw=0.5)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "pnl_inventory_comparison.png"), dpi=150)
    plt.close()
    print("Saved pnl_inventory_comparison.png")

    # --- Plot 2: Action distributions ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (acts, name, color) in enumerate([
        (as_actions, "Avellaneda-Stoikov", "steelblue"),
        (dqn_actions, "DQN", "coral"),
        (ppo_actions, "PPO", "seagreen"),
    ]):
        counts = Counter(acts)
        x = list(range(25))
        y = [counts.get(i, 0) for i in x]
        axes[idx].bar(x, y, color=color, alpha=0.8)
        axes[idx].set_xlabel("Action Index (bid_offset × 5 + ask_offset)")
        axes[idx].set_ylabel("Count")
        axes[idx].set_title(f"{name} Action Distribution")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "action_distributions.png"), dpi=150)
    plt.close()
    print("Saved action_distributions.png")

    # --- Plot 3: Action heatmaps ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (acts, name) in enumerate([
        (as_actions, "Avellaneda-Stoikov"),
        (dqn_actions, "DQN"),
        (ppo_actions, "PPO"),
    ]):
        grid = np.zeros((5, 5))
        for a in acts:
            bid_idx = a // 5
            ask_idx = a % 5
            if 0 <= bid_idx < 5 and 0 <= ask_idx < 5:
                grid[bid_idx, ask_idx] += 1
        grid = grid / max(grid.sum(), 1)

        im = axes[idx].imshow(grid, cmap="YlOrRd", aspect="auto")
        axes[idx].set_xlabel("Ask Offset (ticks)")
        axes[idx].set_ylabel("Bid Offset (ticks)")
        axes[idx].set_xticks(range(5))
        axes[idx].set_xticklabels([1, 2, 3, 4, 5])
        axes[idx].set_yticks(range(5))
        axes[idx].set_yticklabels([1, 2, 3, 4, 5])
        axes[idx].set_title(f"{name}")
        for bi in range(5):
            for ai in range(5):
                axes[idx].text(ai, bi, f"{grid[bi, ai]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=axes[idx], shrink=0.8)

    plt.suptitle("Action Frequency Heatmaps (bid offset × ask offset)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "action_heatmaps.png"), dpi=150)
    plt.close()
    print("Saved action_heatmaps.png")

    # --- Save results ---
    def serialize(stats):
        return {k: float(v) if isinstance(v, (np.floating, float)) else
                   int(v) if isinstance(v, (np.integer, int)) else v
                for k, v in stats.items()}

    all_results = {
        "avellaneda_stoikov": serialize(as_stats),
        "dqn": serialize(dqn_stats),
        "ppo": serialize(ppo_stats),
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved results.json")

    min_len = min(len(as_pnl), len(dqn_pnl), len(ppo_pnl))
    pd.DataFrame({
        "as": as_pnl[:min_len], "dqn": dqn_pnl[:min_len], "ppo": ppo_pnl[:min_len]
    }).to_csv(os.path.join(RESULTS_DIR, "pnl_traces.csv"), index=False)
    pd.DataFrame({
        "as": as_inv[:min_len], "dqn": dqn_inv[:min_len], "ppo": ppo_inv[:min_len]
    }).to_csv(os.path.join(RESULTS_DIR, "inventory_traces.csv"), index=False)
    print("Saved trace data for .rmd report")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    step1_preprocess()

    print("\nLoading preprocessed snapshots...")
    train_data, test_data = load_snapshots(SNAPSHOT_DIR)

    as_stats, as_pnl, as_inv, as_actions, as_grid = step2_baseline(test_data)
    as_grid.to_csv(os.path.join(RESULTS_DIR, "as_grid_search.csv"), index=False)

    _, dqn_stats, dqn_pnl, dqn_inv, dqn_actions = step3_train_dqn(train_data, test_data)
    _, ppo_stats, ppo_pnl, ppo_inv, ppo_actions = step4_train_ppo(train_data, test_data)

    step5_compare(
        as_stats, as_pnl, as_inv, as_actions,
        dqn_stats, dqn_pnl, dqn_inv, dqn_actions,
        ppo_stats, ppo_pnl, ppo_inv, ppo_actions,
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Results saved to {RESULTS_DIR}/")
    print(f"Models saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()

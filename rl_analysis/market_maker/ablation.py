"""
Robustness analysis:
  1. Random agent baseline
  2. Feature ablation: retrain DQN with different feature subsets
"""

import os
import json
import numpy as np
import pandas as pd

from .data_preprocessor import load_snapshots
from .env import MarketMakingEnv
from .baseline import AvellanedaStoikov

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

SNAPSHOT_DIR = "snapshots"
MODEL_DIR = "models"
RESULTS_DIR = "rl_results"
EPISODE_LENGTH = 3600
N_EVAL_EPISODES = 50
DQN_TIMESTEPS = 100_000

ALL_FEATURES = [
    "spread",
    "book_imbalance",
    "ofi_rolling",
    "volatility",
    "trade_intensity",
    "cancel_add_ratio",
]

# "No order flow" = only spread + volatility (price-derived, no order book signal)
PRICE_ONLY_FEATURES = ["spread", "volatility"]

# Features grouped by research question
ABLATION_CONFIGS = {
    "all_features": ALL_FEATURES,
    "price_only": PRICE_ONLY_FEATURES,
    "no_ofi": [f for f in ALL_FEATURES if f != "ofi_rolling"],
    "no_imbalance": [f for f in ALL_FEATURES if f != "book_imbalance"],
    "no_cancel_ratio": [f for f in ALL_FEATURES if f != "cancel_add_ratio"],
    "ofi_only": ["spread", "ofi_rolling", "volatility"],
}


def evaluate_multi(
    agent_name, test_data, n_episodes, model=None, obs_features=None, random_agent=False
):
    records = []
    for ep in range(n_episodes):
        env = MarketMakingEnv(
            data=test_data, episode_length=EPISODE_LENGTH, obs_features=obs_features
        )
        obs, _ = env.reset(seed=ep)
        done = False

        while not done:
            if random_agent:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        stats = env.get_episode_stats()
        stats["episode"] = ep
        stats["agent"] = agent_name
        records.append(stats)

    return pd.DataFrame(records)


def run_random_baseline(test_data):
    print("=" * 60)
    print("RANDOM AGENT BASELINE")
    print("=" * 60)

    df = evaluate_multi("Random", test_data, N_EVAL_EPISODES, random_agent=True)

    pnl = df["total_pnl"]
    print(f"  PnL:      mean={pnl.mean():.2f}, std={pnl.std():.2f}")
    print(f"  Win rate: {(pnl > 0).sum()}/{len(pnl)} ({100 * (pnl > 0).mean():.0f}%)")
    print(f"  Mean |Inv|: {df['final_inventory'].abs().mean():.0f}")
    return df


def run_ablation(train_data, test_data):
    print("\n" + "=" * 60)
    print("FEATURE ABLATION STUDY")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)
    all_results = []

    for config_name, features in ABLATION_CONFIGS.items():
        print(f"\n--- {config_name}: {features} ---")

        model_path = os.path.join(MODEL_DIR, f"dqn_ablation_{config_name}")

        if os.path.exists(model_path + ".zip"):
            print(f"  Loading existing model...")
            model = DQN.load(model_path)
        else:
            print(f"  Training DQN ({DQN_TIMESTEPS} steps)...")
            train_env = Monitor(
                MarketMakingEnv(
                    data=train_data,
                    episode_length=EPISODE_LENGTH,
                    obs_features=features,
                )
            )

            model = DQN(
                "MlpPolicy",
                train_env,
                learning_rate=1e-4,
                buffer_size=50_000,
                learning_starts=1_000,
                batch_size=64,
                gamma=0.99,
                exploration_fraction=0.3,
                exploration_final_eps=0.05,
                target_update_interval=500,
                policy_kwargs={"net_arch": [128, 128]},
                verbose=0,
            )
            model.learn(total_timesteps=DQN_TIMESTEPS)
            model.save(model_path)

        print(f"  Evaluating over {N_EVAL_EPISODES} episodes...")
        df = evaluate_multi(
            config_name, test_data, N_EVAL_EPISODES, model=model, obs_features=features
        )

        pnl = df["total_pnl"]
        print(f"  PnL:      mean={pnl.mean():.2f}, std={pnl.std():.2f}")
        print(
            f"  Win rate: {(pnl > 0).sum()}/{len(pnl)} ({100 * (pnl > 0).mean():.0f}%)"
        )

        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading snapshots...")
    train_data, test_data = load_snapshots(SNAPSHOT_DIR)

    # 1. Random baseline
    random_df = run_random_baseline(test_data)

    # 2. Feature ablation
    ablation_df = run_ablation(train_data, test_data)

    # Combine and save
    combined = pd.concat([random_df, ablation_df], ignore_index=True)
    combined.to_csv(os.path.join(RESULTS_DIR, "ablation_stats.csv"), index=False)

    # Summary table
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    summary = (
        combined.groupby("agent")
        .agg(
            mean_pnl=("total_pnl", "mean"),
            std_pnl=("total_pnl", "std"),
            mean_sharpe=("sharpe", "mean"),
            win_rate=("total_pnl", lambda x: (x > 0).mean()),
            mean_abs_inv=("final_inventory", lambda x: x.abs().mean()),
        )
        .round(2)
    )
    print(summary)

    # T-tests: each ablation vs all_features
    from scipy import stats as sp

    all_feat_pnl = ablation_df[ablation_df["agent"] == "all_features"][
        "total_pnl"
    ].values

    tests = {}
    for name in combined["agent"].unique():
        if name == "all_features":
            continue
        sub_pnl = combined[combined["agent"] == name]["total_pnl"].values
        t, p = sp.ttest_ind(all_feat_pnl, sub_pnl)
        tests[name] = {"t": float(t), "p": float(p)}
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "ns"
        print(f"  all_features vs {name}: t={t:.3f}, p={p:.4f} {sig}")

    # Save summary JSON
    summary_dict = {
        name: {
            "mean_pnl": float(combined[combined["agent"] == name]["total_pnl"].mean()),
            "std_pnl": float(combined[combined["agent"] == name]["total_pnl"].std()),
            "mean_sharpe": float(combined[combined["agent"] == name]["sharpe"].mean()),
            "win_rate": float(
                (combined[combined["agent"] == name]["total_pnl"] > 0).mean()
            ),
            "mean_abs_inv": float(
                combined[combined["agent"] == name]["final_inventory"].abs().mean()
            ),
        }
        for name in combined["agent"].unique()
    }
    summary_dict["t_tests"] = tests

    with open(os.path.join(RESULTS_DIR, "ablation_results.json"), "w") as f:
        json.dump(summary_dict, f, indent=2)

    print(f"\nSaved ablation_stats.csv and ablation_results.json")


if __name__ == "__main__":
    main()

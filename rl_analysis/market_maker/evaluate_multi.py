"""
Multi-episode evaluation for statistical validation.
Runs each agent across N random test episodes and saves per-episode stats.
"""

import os
import json
import numpy as np
import pandas as pd

from .data_preprocessor import load_snapshots
from .env import MarketMakingEnv
from .baseline import AvellanedaStoikov

from stable_baselines3 import DQN, PPO

SNAPSHOT_DIR = "snapshots"
MODEL_DIR = "models"
RESULTS_DIR = "rl_results"
EPISODE_LENGTH = 3600
N_EPISODES = 50


def evaluate_agent_multi(
    agent_name, test_data, n_episodes=N_EPISODES, model=None, as_model=None
):
    """Run agent for n_episodes with different random seeds, collect stats."""
    records = []
    for ep in range(n_episodes):
        env = MarketMakingEnv(data=test_data, episode_length=EPISODE_LENGTH)
        obs, _ = env.reset(seed=ep)
        done = False

        while not done:
            if as_model is not None:
                mid = env.data.iloc[env._step_idx]["mid_price"]
                sigma = env.data.iloc[env._step_idx]["volatility"]
                sigma = sigma if not np.isnan(sigma) and sigma > 0 else 0.001
                t_rem = 1.0 - obs[-1]
                action = as_model.get_action_for_env(
                    mid, env.inventory, sigma, t_rem, env.BID_OFFSETS, env.ASK_OFFSETS
                )
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        stats = env.get_episode_stats()
        stats["episode"] = ep
        stats["agent"] = agent_name
        records.append(stats)

        if (ep + 1) % 10 == 0:
            print(f"  {agent_name}: {ep + 1}/{n_episodes} episodes done")

    return pd.DataFrame(records)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading snapshots...")
    _, test_data = load_snapshots(SNAPSHOT_DIR)

    # Load best AS params
    results = json.load(open(os.path.join(RESULTS_DIR, "results.json")))
    gamma = results["avellaneda_stoikov"].get("gamma", 0.01)
    k = results["avellaneda_stoikov"].get("k", 0.5)

    print(f"\nEvaluating AS (gamma={gamma}, k={k}) over {N_EPISODES} episodes...")
    as_model = AvellanedaStoikov(gamma=gamma, k=k)
    as_df = evaluate_agent_multi("AS", test_data, as_model=as_model)

    print(f"\nEvaluating DQN over {N_EPISODES} episodes...")
    dqn = DQN.load(os.path.join(MODEL_DIR, "dqn_market_maker"))
    dqn_df = evaluate_agent_multi("DQN", test_data, model=dqn)

    print(f"\nEvaluating PPO over {N_EPISODES} episodes...")
    ppo = PPO.load(os.path.join(MODEL_DIR, "ppo_market_maker"))
    ppo_df = evaluate_agent_multi("PPO", test_data, model=ppo)

    all_df = pd.concat([as_df, dqn_df, ppo_df], ignore_index=True)
    all_df.to_csv(os.path.join(RESULTS_DIR, "multi_episode_stats.csv"), index=False)

    # Print summary
    print("\n" + "=" * 70)
    print(f"MULTI-EPISODE EVALUATION ({N_EPISODES} episodes per agent)")
    print("=" * 70)
    for name in ["AS", "DQN", "PPO"]:
        sub = all_df[all_df["agent"] == name]
        pnl = sub["total_pnl"]
        sharpe = sub["sharpe"]
        inv = sub["final_inventory"].abs()
        print(f"\n{name}:")
        print(
            f"  PnL:     mean={pnl.mean():.2f}, std={pnl.std():.2f}, "
            f"95% CI=[{pnl.mean() - 1.96 * pnl.std() / np.sqrt(len(pnl)):.2f}, "
            f"{pnl.mean() + 1.96 * pnl.std() / np.sqrt(len(pnl)):.2f}]"
        )
        print(f"  Sharpe:  mean={sharpe.mean():.2f}, std={sharpe.std():.2f}")
        print(f"  |Inv|:   mean={inv.mean():.0f}, max={inv.max():.0f}")
        wins = (pnl > 0).sum()
        print(f"  Win rate: {wins}/{len(pnl)} ({100 * wins / len(pnl):.0f}%)")

    # Paired t-test: DQN vs AS, PPO vs AS
    from scipy import stats as scipy_stats

    as_pnl = as_df["total_pnl"].values
    dqn_pnl = dqn_df["total_pnl"].values
    ppo_pnl = ppo_df["total_pnl"].values

    t_dqn, p_dqn = scipy_stats.ttest_ind(dqn_pnl, as_pnl)
    t_ppo, p_ppo = scipy_stats.ttest_ind(ppo_pnl, as_pnl)

    print(f"\nStatistical tests (two-sample t-test vs AS):")
    print(
        f"  DQN vs AS: t={t_dqn:.3f}, p={p_dqn:.4f} {'***' if p_dqn < 0.01 else '**' if p_dqn < 0.05 else 'ns'}"
    )
    print(
        f"  PPO vs AS: t={t_ppo:.3f}, p={p_ppo:.4f} {'***' if p_ppo < 0.01 else '**' if p_ppo < 0.05 else 'ns'}"
    )

    # Save test results
    test_results = {
        "n_episodes": N_EPISODES,
        "AS": {
            "pnl_mean": float(as_df["total_pnl"].mean()),
            "pnl_std": float(as_df["total_pnl"].std()),
            "sharpe_mean": float(as_df["sharpe"].mean()),
            "win_rate": float((as_df["total_pnl"] > 0).mean()),
        },
        "DQN": {
            "pnl_mean": float(dqn_df["total_pnl"].mean()),
            "pnl_std": float(dqn_df["total_pnl"].std()),
            "sharpe_mean": float(dqn_df["sharpe"].mean()),
            "win_rate": float((dqn_df["total_pnl"] > 0).mean()),
            "vs_as_t": float(t_dqn),
            "vs_as_p": float(p_dqn),
        },
        "PPO": {
            "pnl_mean": float(ppo_df["total_pnl"].mean()),
            "pnl_std": float(ppo_df["total_pnl"].std()),
            "sharpe_mean": float(ppo_df["sharpe"].mean()),
            "win_rate": float((ppo_df["total_pnl"] > 0).mean()),
            "vs_as_t": float(t_ppo),
            "vs_as_p": float(p_ppo),
        },
    }
    with open(os.path.join(RESULTS_DIR, "multi_episode_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\nSaved multi_episode_stats.csv and multi_episode_results.json")


if __name__ == "__main__":
    main()

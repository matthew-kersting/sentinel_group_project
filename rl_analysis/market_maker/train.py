"""
Training script for RL market making agents.

Usage:
    python -m market_maker.train --algo dqn --timesteps 100000
    python -m market_maker.train --algo ppo --timesteps 500000
"""

import argparse
import os
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from .data_preprocessor import load_snapshots
from .env import MarketMakingEnv
from .baseline import run_baseline, grid_search_baseline


SNAPSHOT_DIR = "snapshots"
MODEL_DIR = "models"
LOG_DIR = "logs"


def make_env(data: pd.DataFrame, episode_length: int = 3600) -> MarketMakingEnv:
    """Create a monitored market making environment."""
    env = MarketMakingEnv(
        data=data,
        max_inventory=1000,
        inventory_penalty=0.01,
        fill_size=100,
        episode_length=episode_length,
    )
    return Monitor(env)


def train_dqn(train_data, test_data, timesteps=100_000, episode_length=3600):
    """Train a DQN agent."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    train_env = make_env(train_data, episode_length)
    eval_env = make_env(test_data, episode_length)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_DIR, "dqn_best"),
        log_path=LOG_DIR,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
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
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    print(f"Training DQN for {timesteps:,} timesteps...")
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    model.save(os.path.join(MODEL_DIR, "dqn_final"))
    print("DQN training complete.")

    return model


def train_ppo(train_data, test_data, timesteps=500_000, episode_length=3600):
    """Train a PPO agent."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    train_env = make_env(train_data, episode_length)
    eval_env = make_env(test_data, episode_length)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_DIR, "ppo_best"),
        log_path=LOG_DIR,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    print(f"Training PPO for {timesteps:,} timesteps...")
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    model.save(os.path.join(MODEL_DIR, "ppo_final"))
    print("PPO training complete.")

    return model


def evaluate_agent(model, env, n_episodes=10) -> pd.DataFrame:
    """Evaluate a trained agent over multiple episodes."""
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        stats = env.get_episode_stats()
        stats["episode"] = ep
        stats["total_reward"] = total_reward
        results.append(stats)

    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "ppo", "baseline"], default="dqn")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--episode-length", type=int, default=3600)
    parser.add_argument("--snapshot-dir", default=SNAPSHOT_DIR)
    args = parser.parse_args()

    print("Loading preprocessed snapshots...")
    train_data, test_data = load_snapshots(args.snapshot_dir)

    if args.algo == "baseline":
        print("\nRunning Avellaneda-Stoikov baseline grid search...")
        env = MarketMakingEnv(test_data, episode_length=args.episode_length)
        results = grid_search_baseline(env)
        print("\n", results.sort_values("total_pnl", ascending=False).head(10))

    elif args.algo == "dqn":
        model = train_dqn(train_data, test_data, args.timesteps, args.episode_length)
        print("\nEvaluating DQN on test data...")
        eval_env = MarketMakingEnv(test_data, episode_length=args.episode_length)
        results = evaluate_agent(model, eval_env)
        print(results.describe())

    elif args.algo == "ppo":
        model = train_ppo(train_data, test_data, args.timesteps, args.episode_length)
        print("\nEvaluating PPO on test data...")
        eval_env = MarketMakingEnv(test_data, episode_length=args.episode_length)
        results = evaluate_agent(model, eval_env)
        print(results.describe())

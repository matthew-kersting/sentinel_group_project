"""
Avellaneda-Stoikov (2008) analytical market making baseline.

Computes optimal bid/ask quotes based on:
  - Current mid price
  - Inventory position
  - Estimated volatility
  - Risk aversion parameter gamma
  - Order arrival intensity k

Reference: Avellaneda & Stoikov, "High-frequency trading in a limit order book",
           Quantitative Finance, 2008.
"""

import numpy as np
import pandas as pd


class AvellanedaStoikov:
    """
    Optimal quote placement following the Avellaneda-Stoikov model.

    reservation_price = s - q * gamma * sigma^2 * (T - t)
    optimal_spread = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)

    bid = reservation_price - spread / 2
    ask = reservation_price + spread / 2
    """

    def __init__(
        self,
        gamma: float = 0.1,
        k: float = 1.5,
        T: float = 1.0,
        tick_size: float = 0.01,
    ):
        """
        Args:
            gamma: Risk aversion parameter. Higher = more aggressive inventory management.
            k: Order arrival intensity parameter. Higher = more orders expected.
            T: Time horizon (1.0 = one trading day).
            tick_size: Minimum price increment.
        """
        self.gamma = gamma
        self.k = k
        self.T = T
        self.tick_size = tick_size

    def get_quotes(
        self,
        mid_price: float,
        inventory: int,
        sigma: float,
        time_remaining: float,
    ) -> tuple[float, float]:
        """
        Compute optimal bid and ask quotes.

        Args:
            mid_price: Current mid price.
            inventory: Current inventory (positive = long).
            sigma: Estimated volatility (std of returns).
            time_remaining: Fraction of trading day remaining [0, 1].

        Returns:
            (bid_price, ask_price)
        """
        dt = max(time_remaining * self.T, 1e-6)

        # Reservation price: skew away from inventory
        reservation = mid_price - inventory * self.gamma * (sigma ** 2) * dt

        # Optimal spread
        spread = self.gamma * (sigma ** 2) * dt + (2 / self.gamma) * np.log(1 + self.gamma / self.k)

        # Ensure minimum spread of 1 tick
        spread = max(spread, self.tick_size)

        bid = reservation - spread / 2
        ask = reservation + spread / 2

        # Round to tick size
        bid = np.floor(bid / self.tick_size) * self.tick_size
        ask = np.ceil(ask / self.tick_size) * self.tick_size

        return bid, ask

    def get_action_for_env(
        self,
        mid_price: float,
        inventory: int,
        sigma: float,
        time_remaining: float,
        bid_offsets: np.ndarray,
        ask_offsets: np.ndarray,
        tick_size: float = 0.01,
    ) -> int:
        """
        Convert AS optimal quotes into a discrete action for the Gym env.

        Maps the continuous optimal quote to the nearest discrete offset.
        """
        bid, ask = self.get_quotes(mid_price, inventory, sigma, time_remaining)

        bid_offset_ticks = max(1, round((mid_price - bid) / tick_size))
        ask_offset_ticks = max(1, round((ask - mid_price) / tick_size))

        # Clip to valid range
        bid_idx = np.argmin(np.abs(bid_offsets - bid_offset_ticks))
        ask_idx = np.argmin(np.abs(ask_offsets - ask_offset_ticks))

        return int(bid_idx * len(ask_offsets) + ask_idx)


def run_baseline(env, gamma: float = 0.1, k: float = 1.5, seed: int = None) -> dict:
    """
    Run the Avellaneda-Stoikov baseline on the environment for one episode.

    Returns episode stats.
    """
    model = AvellanedaStoikov(gamma=gamma, k=k)

    obs, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0

    while not done:
        # Extract state from observation
        mid_price = env.data.iloc[env._step_idx]["mid_price"]
        sigma = env.data.iloc[env._step_idx]["volatility"]
        if np.isnan(sigma) or sigma == 0:
            sigma = 0.001

        time_remaining = 1.0 - obs[-1]  # last obs element is time fraction

        action = model.get_action_for_env(
            mid_price=mid_price,
            inventory=env.inventory,
            sigma=sigma,
            time_remaining=time_remaining,
            bid_offsets=env.BID_OFFSETS,
            ask_offsets=env.ASK_OFFSETS,
        )

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    stats = env.get_episode_stats()
    stats["total_reward"] = total_reward
    return stats


def grid_search_baseline(env, gammas=None, ks=None) -> pd.DataFrame:
    """
    Grid search over gamma and k to find best AS parameters.
    """
    if gammas is None:
        gammas = [0.01, 0.05, 0.1, 0.5, 1.0]
    if ks is None:
        ks = [0.5, 1.0, 1.5, 2.0, 5.0]

    results = []
    for g in gammas:
        for k in ks:
            stats = run_baseline(env, gamma=g, k=k, seed=42)
            stats["gamma"] = g
            stats["k"] = k
            results.append(stats)
            print(f"  gamma={g:.2f}, k={k:.1f}: PnL={stats['total_pnl']:.2f}, "
                  f"Sharpe={stats['sharpe']:.2f}, Trades={stats['n_trades']}")

    return pd.DataFrame(results)

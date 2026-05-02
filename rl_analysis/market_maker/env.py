"""
Gymnasium environment for market making on pre-processed MBO snapshots.

The agent acts as a market maker: at each timestep (1 second), it decides
where to place bid and ask quotes relative to the current mid price.
Fills are determined by checking whether real market trades during that
interval would have crossed the agent's quotes.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class MarketMakingEnv(gym.Env):
    """
    Observation space:
        [spread, book_imbalance, ofi_rolling, volatility, trade_intensity,
         cancel_add_ratio, inventory_normalized, unrealized_pnl_normalized,
         time_fraction]

    Action space (discrete, 25 actions):
        5 bid offsets x 5 ask offsets (in ticks from mid price)
        bid_offset in {1, 2, 3, 4, 5} ticks below mid
        ask_offset in {1, 2, 3, 4, 5} ticks above mid

    Reward:
        PnL from completed round-trips + spread capture
        - penalty for holding inventory
    """

    metadata = {"render_modes": []}

    # Discrete offsets in ticks (1 tick = $0.01)
    BID_OFFSETS = np.array([1, 2, 3, 4, 5])
    ASK_OFFSETS = np.array([1, 2, 3, 4, 5])
    TICK_SIZE = 0.01

    # Environment parameters
    MAX_INVENTORY = 1000  # max shares to hold
    INVENTORY_PENALTY = 0.01  # penalty coefficient per share per step
    FILL_SIZE = 100  # shares per fill (1 round lot)

    # Feature columns used for observation
    OBS_FEATURES = [
        "spread",
        "book_imbalance",
        "ofi_rolling",
        "volatility",
        "trade_intensity",
        "cancel_add_ratio",
    ]

    def __init__(
        self,
        data: pd.DataFrame,
        max_inventory: int = 1000,
        inventory_penalty: float = 0.01,
        fill_size: int = 100,
        episode_length: int | None = None,
        obs_features: list | None = None,
    ):
        super().__init__()

        if obs_features is not None:
            self.OBS_FEATURES = list(obs_features)

        self.raw_data = data.copy().reset_index(drop=True)
        self._prepare_data()

        self.max_inventory = max_inventory
        self.inventory_penalty = inventory_penalty
        self.fill_size = fill_size

        self.episode_length = episode_length

        n_bid = len(self.BID_OFFSETS)
        n_ask = len(self.ASK_OFFSETS)
        self.action_space = spaces.Discrete(n_bid * n_ask)

        n_obs = len(self.OBS_FEATURES) + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )

        self._step_idx = 0
        self._start_idx = 0
        self._end_idx = len(self.data) - 1
        self.inventory = 0
        self.cash = 0.0
        self.total_pnl = 0.0
        self.trades_executed = 0

    def _prepare_data(self):
        """Clean and normalize feature data."""
        df = self.raw_data.copy()

        df["spread"] = df["spread"].ffill().fillna(0.01)
        df["book_imbalance"] = df["book_imbalance"].fillna(0)
        df["ofi_rolling"] = df["ofi_rolling"].fillna(0)
        df["volatility"] = df["volatility"].ffill().fillna(0)
        df["trade_intensity"] = df["trade_intensity"].fillna(0)
        df["cancel_add_ratio"] = df["cancel_add_ratio"].fillna(0).clip(upper=10)

        # Drop rows with no valid mid price
        df = df.dropna(subset=["mid_price"]).reset_index(drop=True)

        # Compute normalization stats
        self._feature_means = {}
        self._feature_stds = {}
        for col in self.OBS_FEATURES:
            self._feature_means[col] = df[col].mean()
            self._feature_stds[col] = df[col].std()
            if self._feature_stds[col] == 0:
                self._feature_stds[col] = 1.0

        self.data = df

    def _decode_action(self, action: int) -> tuple[int, int]:
        """Convert action index to (bid_offset_ticks, ask_offset_ticks)."""
        n_ask = len(self.ASK_OFFSETS)
        bid_idx = action // n_ask
        ask_idx = action % n_ask
        return int(self.BID_OFFSETS[bid_idx]), int(self.ASK_OFFSETS[ask_idx])

    def _get_observation(self) -> np.ndarray:
        row = self.data.iloc[self._step_idx]

        features = []
        for col in self.OBS_FEATURES:
            val = (row[col] - self._feature_means[col]) / self._feature_stds[col]
            features.append(val)

        # Normalized inventory: [-1, 1]
        inv_norm = self.inventory / self.max_inventory
        features.append(inv_norm)

        # Unrealized PnL normalized by price
        mid = row["mid_price"]
        unrealized = self.inventory * mid
        features.append(unrealized / (mid * self.max_inventory) if mid > 0 else 0)

        # Time fraction within episode
        if self.episode_length:
            t_frac = (self._step_idx - self._start_idx) / self.episode_length
        else:
            t_frac = (self._step_idx - self._start_idx) / max(
                1, self._end_idx - self._start_idx
            )
        features.append(t_frac)

        return np.array(features, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.episode_length and len(self.data) > self.episode_length:
            max_start = len(self.data) - self.episode_length
            self._start_idx = self.np_random.integers(0, max_start)
            self._end_idx = self._start_idx + self.episode_length
        else:
            self._start_idx = 0
            self._end_idx = len(self.data) - 1

        self._step_idx = self._start_idx
        self.inventory = 0
        self.cash = 0.0
        self.total_pnl = 0.0
        self.trades_executed = 0
        self._pnl_history = []
        self._inventory_history = []

        obs = self._get_observation()
        return obs, {}

    def step(self, action: int):
        bid_offset, ask_offset = self._decode_action(action)

        row = self.data.iloc[self._step_idx]
        mid = row["mid_price"]

        # Agent's quote prices
        bid_price = mid - bid_offset * self.TICK_SIZE
        ask_price = mid + ask_offset * self.TICK_SIZE

        # Check for fills using real trade data from the interval
        bid_filled = False
        ask_filled = False

        # Agent's bid fills if a market sell order traded at or below our bid
        min_sell = row.get("min_sell_trade_price", np.nan)
        if not np.isnan(min_sell) and min_sell <= bid_price:
            bid_filled = True

        # Agent's ask fills if a market buy order traded at or above our ask
        max_buy = row.get("max_buy_trade_price", np.nan)
        if not np.isnan(max_buy) and max_buy >= ask_price:
            ask_filled = True

        # Enforce inventory limits
        if self.inventory >= self.max_inventory:
            bid_filled = False
        if self.inventory <= -self.max_inventory:
            ask_filled = False

        # Execute fills
        step_pnl = 0.0
        if bid_filled:
            self.inventory += self.fill_size
            self.cash -= bid_price * self.fill_size
            self.trades_executed += 1

        if ask_filled:
            self.inventory -= self.fill_size
            self.cash += ask_price * self.fill_size
            self.trades_executed += 1

        # Mark-to-market PnL
        next_idx = min(self._step_idx + 1, self._end_idx)
        next_mid = self.data.iloc[next_idx]["mid_price"]
        mark_to_market = self.cash + self.inventory * next_mid
        step_pnl = mark_to_market - self.total_pnl
        self.total_pnl = mark_to_market

        # Reward: PnL change minus inventory penalty
        inventory_cost = self.inventory_penalty * abs(self.inventory) * self.TICK_SIZE
        reward = step_pnl - inventory_cost

        self._pnl_history.append(self.total_pnl)
        self._inventory_history.append(self.inventory)

        # Advance
        self._step_idx += 1
        terminated = self._step_idx >= self._end_idx
        truncated = False

        # Terminal: liquidate remaining inventory at mid price
        if terminated and self.inventory != 0:
            liquidation_cost = (
                abs(self.inventory) * row["spread"] / 2
                if not np.isnan(row["spread"])
                else 0
            )
            reward -= liquidation_cost

        obs = (
            self._get_observation()
            if not terminated
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )

        info = {
            "inventory": self.inventory,
            "total_pnl": self.total_pnl,
            "trades": self.trades_executed,
            "bid_filled": bid_filled,
            "ask_filled": ask_filled,
            "mid_price": mid,
            "bid_price": bid_price,
            "ask_price": ask_price,
        }

        return obs, reward, terminated, truncated, info

    def get_episode_stats(self) -> dict:
        """Summary stats for the completed episode."""
        pnl = np.array(self._pnl_history) if self._pnl_history else np.array([0])
        returns = np.diff(pnl)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(23400)

        return {
            "total_pnl": self.total_pnl,
            "sharpe": sharpe,
            "max_drawdown": np.min(pnl - np.maximum.accumulate(pnl)),
            "n_trades": self.trades_executed,
            "final_inventory": self.inventory,
            "max_inventory": max(abs(i) for i in self._inventory_history)
            if self._inventory_history
            else 0,
        }

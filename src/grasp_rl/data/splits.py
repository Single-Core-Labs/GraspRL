"""Train/validation/test split utilities for demonstration data."""

import math
import random
from typing import List, Tuple
from grasp_rl.data.schemas import TrajectorySchema, StateActionSchema


def train_val_test_split(
    episodes: List,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[List, List, List]:
    """Split episodes into train, validation, and test sets.

    Splits are episode-level (never splits within an episode) and seeded
    for reproducibility. Source data is never mutated.

    Args:
        episodes: List of episodes or trajectories to split.
        seed: Random seed for reproducibility.
        train_ratio: Fraction of data for training (default 0.8).
        val_ratio: Fraction of data for validation (default 0.1).
        test_ratio: Fraction of data for testing (default 0.1).

    Returns:
        Tuple of (train_episodes, val_episodes, test_episodes).

    Raises:
        ValueError: If ratios don't sum to 1.0.
    """
    # Validate ratios
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6):
        raise ValueError(
            f"Ratios must sum to 1.0: train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

    # Set seed for reproducibility
    random.seed(seed)

    # Create a copy to avoid mutating input
    episodes_copy = list(episodes)

    # Shuffle episodes
    random.shuffle(episodes_copy)

    # Calculate split indices
    n = len(episodes_copy)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # Split
    train = episodes_copy[:n_train]
    val = episodes_copy[n_train : n_train + n_val]
    test = episodes_copy[n_train + n_val :]

    return train, val, test

"""Tests for data split utilities."""

import pytest
from grasp_rl.data.splits import train_val_test_split


def test_train_val_test_split_basic():
    """Test basic split with default ratios."""
    episodes = list(range(100))
    train, val, test = train_val_test_split(episodes, seed=42)

    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10
    assert len(train) + len(val) + len(test) == 100


def test_train_val_test_split_reproducibility():
    """Test that splits are reproducible with same seed."""
    episodes = list(range(100))
    train1, val1, test1 = train_val_test_split(episodes, seed=42)
    train2, val2, test2 = train_val_test_split(episodes, seed=42)

    assert train1 == train2
    assert val1 == val2
    assert test1 == test2


def test_train_val_test_split_different_seeds():
    """Test that different seeds produce different splits."""
    episodes = list(range(100))
    train1, _, _ = train_val_test_split(episodes, seed=42)
    train2, _, _ = train_val_test_split(episodes, seed=99)

    assert train1 != train2


def test_train_val_test_split_custom_ratios():
    """Test split with custom ratios."""
    episodes = list(range(100))
    train, val, test = train_val_test_split(
        episodes, seed=42, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
    )

    assert len(train) == 70
    assert len(val) == 20
    assert len(test) == 10


def test_train_val_test_split_invalid_ratios():
    """Test that invalid ratios raise ValueError."""
    episodes = list(range(100))
    with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
        train_val_test_split(episodes, seed=42, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


def test_train_val_test_split_no_mutation():
    """Test that input data is not mutated."""
    episodes = list(range(100))
    original_order = list(episodes)
    train_val_test_split(episodes, seed=42)

    assert episodes == original_order


def test_train_val_test_split_empty_data():
    """Test split with empty data."""
    episodes = []
    train, val, test = train_val_test_split(episodes, seed=42)

    assert len(train) == 0
    assert len(val) == 0
    assert len(test) == 0


def test_train_val_test_split_single_episode():
    """Test split with single episode."""
    episodes = [1]
    train, val, test = train_val_test_split(episodes, seed=42)

    # With 1 item and 0.8/0.1/0.1 ratios, it should go to train
    assert len(train) == 1
    assert len(val) == 0
    assert len(test) == 0

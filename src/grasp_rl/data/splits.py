"""Train/validation/test split utilities."""


def train_val_test_split(episodes: list, seed: int = 42) -> tuple:
    """Split episodes into train, validation, and test sets."""
    return episodes[:80], episodes[80:90], episodes[90:]

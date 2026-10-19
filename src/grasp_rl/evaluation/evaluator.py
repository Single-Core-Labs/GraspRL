"""Policy evaluation utilities."""


def evaluate_policy(checkpoint_path: str, episodes: int = 100) -> dict:
    """Evaluate a trained policy."""
    return {"success_rate": 0.0, "mean_error": 0.0}

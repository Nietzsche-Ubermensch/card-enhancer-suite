import math
from typing import Union


def calculate_truth_reward(t_star: Union[float, "torch.Tensor"]) -> Union[float, "torch.Tensor"]:
    """
    Piecewise truthfulness reward.

    Negative t_star  -> tanh(t_star) + 2.0  (range: (1.0, 2.0))
    Non-negative     -> 1.5 + sigmoid(t_star) (range: [2.0, 2.5))

    Continuous at 0; not differentiable there (left derivative = 1.0,
    right derivative = 0.25).
    """
    try:
        import torch
        if isinstance(t_star, torch.Tensor):
            neg = torch.tanh(t_star) + 2.0
            pos = 1.5 + torch.sigmoid(t_star)
            return torch.where(t_star < 0, neg, pos)
    except ImportError:
        pass

    if t_star < 0:
        return math.tanh(t_star) + 2.0
    return 1.5 + (1.0 / (1.0 + math.exp(-t_star)))


def quality_score_from_metadata(metadata: dict) -> float:
    """
    Map extracted card metadata to a scalar quality score,
    then feed it into the truth reward.
    """
    score = 0.0
    if metadata.get("subjectName"):
        score += 0.3
    if metadata.get("cardNumber"):
        score += 0.2
    if metadata.get("manufacturer"):
        score += 0.2
    if metadata.get("year"):
        score += 0.15
    if metadata.get("stats"):
        score += 0.15

    # Normalize to roughly [-2, 2] for the reward function
    t_star = (score - 0.5) * 4.0
    return calculate_truth_reward(t_star)

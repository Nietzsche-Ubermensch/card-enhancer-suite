import math

import pytest

from card_enhancer.rewards import calculate_truth_reward, quality_score_from_metadata


def test_negative_branch_range():
    assert 1.0 < calculate_truth_reward(-10) < 2.0
    assert 1.0 < calculate_truth_reward(-0.5) < 2.0


def test_positive_branch_range():
    assert 2.0 <= calculate_truth_reward(0) < 2.5
    assert 2.0 <= calculate_truth_reward(10) < 2.5


def test_continuous_at_zero():
    left = calculate_truth_reward(-1e-9)
    right = calculate_truth_reward(0.0)
    assert left == pytest.approx(right, abs=1e-6)
    assert right == pytest.approx(2.0)


def test_exact_values():
    assert calculate_truth_reward(-2.0) == pytest.approx(math.tanh(-2.0) + 2.0)
    assert calculate_truth_reward(2.0) == pytest.approx(1.5 + 1 / (1 + math.exp(-2)))


def test_quality_score_full_metadata():
    metadata = {
        "subjectName": "Alex Windsor",
        "cardNumber": "3",
        "manufacturer": "Upper Deck",
        "year": 2026,
        "stats": {"height": "5'5\"", "from": "Norwich, England"},
    }
    # score = 1.0 -> t_star = 2.0 -> 1.5 + sigmoid(2.0)
    expected = 1.5 + 1 / (1 + math.exp(-2.0))
    assert quality_score_from_metadata(metadata) == pytest.approx(expected)


def test_quality_score_empty_metadata():
    # score = 0.0 -> t_star = -2.0 -> tanh(-2) + 2
    assert quality_score_from_metadata({}) == pytest.approx(math.tanh(-2.0) + 2.0)


def test_quality_score_bounds():
    assert 1.0 < quality_score_from_metadata({}) < 2.0
    full = {"subjectName": "x", "cardNumber": "1", "manufacturer": "y",
            "year": 2026, "stats": {"a": 1}}
    assert 2.0 <= quality_score_from_metadata(full) < 2.5


def test_torch_tensor_input():
    torch = pytest.importorskip("torch")
    t = torch.tensor([-1.0, 0.0, 1.0])
    out = calculate_truth_reward(t)
    assert torch.allclose(out[0], torch.tensor(math.tanh(-1.0) + 2.0))
    assert torch.allclose(out[1], torch.tensor(2.0))
    assert torch.allclose(out[2], torch.tensor(1.5 + torch.sigmoid(torch.tensor(1.0))))

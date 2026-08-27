"""
End-to-end smoke test: train a tiny ELAN for one epoch on synthetic cards.
"""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # noqa: F841
yaml = pytest.importorskip("yaml")  # noqa: F841
from PIL import Image  # noqa: E402

from elan.train import train  # noqa: E402


def _make_card(path: Path, size=(64, 64), seed=0):
    import random
    random.seed(seed)
    img = Image.new("RGB", size)
    px = img.load()
    for y in range(size[1]):
        for x in range(size[0]):
            px[x, y] = (
                (x * 7 + seed * 13) % 256,
                (y * 5 + seed * 29) % 256,
                ((x + y) * 3 + seed * 7) % 256,
            )
    img.save(path)


def test_train_smoke(tmp_path):
    train_dir = tmp_path / "train"
    valid_dir = tmp_path / "valid"
    train_dir.mkdir()
    valid_dir.mkdir()
    for i in range(6):
        _make_card(train_dir / f"card_{i}.png", seed=i)
    for i in range(2):
        _make_card(valid_dir / f"val_{i}.png", seed=100 + i)

    cfg = {
        "model": "elan",
        "scale": 2,
        "gt_size": 32,
        "batch_size": 2,
        "epochs": 1,
        "lr": 0.001,
        "decays": [],
        "gamma": 0.5,
        "log_every": 1,
        "test_every": 1,
        "num_workers": 0,
        "threads": 1,
        "gpu_ids": [],
        "dataroot": str(train_dir),
        "valid_dataroots": {"cards_val": str(valid_dir)},
        "use_hflip": True,
        "use_rot": True,
        "log_path": str(tmp_path / "experiments"),
        "experiment_name": "smoke",
        "channels": 24,
        "n_blocks": 2,
        "num_heads": 3,
        "n_groups": 4,
    }
    ckpt = train(cfg)
    assert ckpt.exists()
    state = torch.load(ckpt, map_location="cpu")
    assert "model_state_dict" in state
    assert state["epoch"] == 1
    assert (ckpt.parent / "checkpoint_best.pt").exists()
